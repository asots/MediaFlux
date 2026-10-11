"""Browser authority lifecycle through real ASGI and the owner transaction."""
import json
import sqlite3

import pytest

from tests.test_setup_bootstrap import cold_run


@pytest.mark.parametrize('mode',['expiry','invalid_concurrent','disconnect','leftover','rate','oversize_slow'])
def test_browser_authority_lifecycle(tmp_path, mode):
    def action():
        import asyncio
        import re
        import threading
        import uvicorn
        import httpx
        from fastapi.testclient import TestClient
        from app.cli import main
        from app.modules import setup_bootstrap as setup
        observed=[]
        failures=[]
        entered=threading.Event()
        release=threading.Event()
        original_sync=setup._sync_directory
        def sync(path):
            if mode in ('invalid_concurrent','disconnect') and path==tmp_path/'db' and (path/'mediaflux.db').exists():
                entered.set()
                assert release.wait(8), 'owner release deadline'
            original_sync(path)
        setup._sync_directory=sync
        if mode=='expiry':
            setup.TTL=0
        grant=tmp_path/('grant-new' if mode=='leftover' else 'grant')
        origin='http://127.0.0.1:1258'
        def serve(server):
            try:
                with TestClient(server.config.app,base_url=origin) as client:
                    page=client.get('/setup')
                    fields={'username':'operator','password':'test-bootstrap-password','password_confirm':'test-bootstrap-password','csrf_token':re.search(r'name="csrf_token" value="([^"]+)"',page.text).group(1),'setup_code':grant.read_text()}
                    def post(data):
                        return client.post('/setup',data=data,headers={'Origin':origin}).status_code
                    if mode=='expiry':
                        observed.append(post(fields))
                        assert not (tmp_path/'db/mediaflux.db').exists()
                    elif mode=='leftover':
                        observed.append(post(dict(fields,setup_code=(tmp_path/'grant').read_text())))
                        assert not (tmp_path/'db/mediaflux.db').exists()
                        observed.append(post(fields))
                    elif mode=='rate':
                        for _ in range(11):
                            observed.append(post(dict(fields,setup_code='invalid')))
                        assert not (tmp_path/'db/mediaflux.db').exists()
                    elif mode=='invalid_concurrent':
                        observed.append(post(dict(fields,password_confirm='mismatch')))
                        assert not (tmp_path/'db/mediaflux.db').exists()
                        answers=[]
                        request=threading.Thread(target=lambda:answers.append(post(fields)))
                        request.start()
                        try:
                            assert entered.wait(8)
                            observed.append(post(fields))
                        finally:
                            release.set()
                            request.join(timeout=8)
                        assert not request.is_alive()
                        observed.extend(answers)
                    elif mode=='disconnect':
                        async def disconnect():
                            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.config.app),base_url=origin) as async_client:
                                task=asyncio.create_task(async_client.post('/setup',data=fields,headers={'Origin':origin}))
                                assert await asyncio.to_thread(entered.wait,8)
                                task.cancel()
                                try:
                                    await task
                                except asyncio.CancelledError:
                                    observed.append('cancelled-request')
                                finally:
                                    release.set()
                        asyncio.run(disconnect())
                    else:
                        observed.append(client.post('/setup',content=b'x'*65537,headers={'Origin':origin}).status_code)
                        async def slow_request():
                            async def body():
                                yield b'a='
                                await asyncio.sleep(12)
                                yield b'b'
                            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.config.app),base_url=origin) as async_client:
                                result=await async_client.post('/setup',content=body(),headers={'Origin':origin})
                                observed.append(result.status_code)
                        asyncio.run(slow_request())
                        assert not (tmp_path/'db/mediaflux.db').exists()
                        observed.append(post(fields))
            except BaseException as exc:
                failures.append(type(exc).__name__)
                release.set()
        uvicorn.Server.run=serve
        result=main(['initialize','--data-dir',str(tmp_path),'--confirm-new-installation','--expect-env-sha256','absent','--web','--grant-file',str(grant)])
        return [result,observed,failures]
    if mode=='leftover':
        def expire():
            import uvicorn
            from app.cli import main
            from app.modules import setup_bootstrap as setup
            setup.TTL=0
            uvicorn.Server.run=lambda server:None
            return main(['initialize','--data-dir',str(tmp_path),'--confirm-new-installation','--expect-env-sha256','absent','--web','--grant-file',str(tmp_path/'grant')])
        assert cold_run(tmp_path,expire)=={'result':2}
    expected={
        'expiry':[2,[409],[]],
        'invalid_concurrent':[0,[400,409,200],[]],
        'disconnect':[0,['cancelled-request'],[]],
        'leftover':[0,[403,200],[]],
        'rate':[2,[403]*10+[429],[]],
        'oversize_slow':[0,[400,400,200],[]],
    }
    assert cold_run(tmp_path,action)=={'result':expected[mode]}
    if mode in ('expiry','rate'):
        assert not (tmp_path/'db/mediaflux.db').exists()
        assert not (tmp_path/'config/user.env').exists()
    else:
        with sqlite3.connect(tmp_path/'db/mediaflux.db') as conn:
            assert json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])['phase']=='stable'
        def login():
            import re
            from fastapi.testclient import TestClient
            from app.main import create_app
            with TestClient(create_app()) as client:
                page=client.get('/login')
                token=re.search(r'name="csrf_token" value="([^"]+)"',page.text).group(1)
                response=client.post('/login',data={'username':'operator','password':'test-bootstrap-password','csrf_token':token},follow_redirects=False)
                return response.status_code,client.get('/readyz').status_code
        assert cold_run(tmp_path,login)=={'result':[302,200]}
