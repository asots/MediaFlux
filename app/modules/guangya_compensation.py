"""光鸭移动/改名的共同补偿边界：响应异常不等于服务端没有提交。"""

from __future__ import annotations

from app.clients.guangya import GuangYaFile


class GuangYaCompensationError(RuntimeError):
    """恢复无法确认；调用方必须持久化人工核验状态，不能宣称已回滚。"""

    def __init__(self, message: str, snapshot: GuangYaFile | None = None):
        super().__init__(message)
        self.snapshot = snapshot


def restore_guangya_file(client, expected: GuangYaFile) -> GuangYaFile:
    """根据实际对象状态恢复名字/父目录，并复核最终状态；不删除任何对象。

    不采信调用者在异常前留下的 current 变量。状态不可读时拒绝盲写；
    补偿本身超时也保留失败语义，交由现有日志/任务表达人工核验。
    """

    def read() -> GuangYaFile:
        try:
            remote = client.file_info(expected.file_id)
        except Exception as exc:
            raise GuangYaCompensationError("补偿状态不可读取，必须人工核验") from exc
        if (
            not isinstance(remote, GuangYaFile)
            or remote.is_dir
            or remote.file_id != expected.file_id
        ):
            raise GuangYaCompensationError("补偿对象身份不可确认，必须人工核验")
        if (expected.size and remote.size != expected.size) or (
            expected.etag and remote.etag != expected.etag
        ):
            raise GuangYaCompensationError(
                "补偿对象内容身份已变化，必须人工核验", remote
            )
        return remote

    if not expected.file_id or not expected.name or not expected.parent_id:
        raise GuangYaCompensationError("补偿目标快照不完整，必须人工核验")
    remote = read()
    changed = False
    try:
        if remote.name != expected.name:
            if client.rename(expected.file_id, expected.name) is False:
                raise RuntimeError("provider returned false")
            changed = True
        if remote.parent_id != expected.parent_id:
            if client.move([expected.file_id], expected.parent_id) is False:
                raise RuntimeError("provider returned false")
            changed = True
    except Exception as exc:
        try:
            latest = read()
        except GuangYaCompensationError:
            latest = None
        raise GuangYaCompensationError(
            "云端补偿请求失败，必须人工核验", latest
        ) from exc
    verified = read() if changed else remote
    if verified.name != expected.name or verified.parent_id != expected.parent_id:
        raise GuangYaCompensationError("云端补偿结果不一致，必须人工核验", verified)
    return verified
