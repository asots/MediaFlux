from __future__ import annotations

from app.clients.http import HttpError


class IndexerError(HttpError):
    """Indexer domain error with a stable public code and safe message."""

    code = "indexer_error"
    default_public_message = "资源索引服务暂不可用"


class IndexerValidationError(IndexerError, ValueError):
    code = "validation_error"
    default_public_message = "搜索参数无效"


class IndexerQueryRejected(IndexerError):
    code = "query_rejected"
    default_public_message = "关键词长度或字符不受该站点支持，请换用完整片名或别名"


class IndexerChallengeRequired(IndexerError):
    code = "challenge_required"
    default_public_message = "站点要求人机验证，本次未完成检索"


class IndexerUnavailable(IndexerError):
    code = "unavailable"
    default_public_message = "索引站点暂不可用"


class IndexerTimeout(IndexerError):
    code = "timeout"
    default_public_message = "索引站点响应超时"


class IndexerRateLimited(IndexerError):
    code = "rate_limited"
    default_public_message = "索引站点请求过于频繁"


class IndexerResultNotFound(IndexerError, LookupError):
    code = "result_not_found"
    default_public_message = "资源结果不存在"


class IndexerResultExpired(IndexerError, LookupError):
    code = "result_expired"
    default_public_message = "资源结果已过期"
