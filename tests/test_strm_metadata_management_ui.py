"""复用现有 DOM 桩执行真实 STRM 脚本，验证关闭态清除及刷新保留。"""

from __future__ import annotations

import textwrap
import unittest

from tests.test_strm_retry_progress import StrmFailureUiJavascriptTests as _UIHarness


class MetadataQueueUiTests(unittest.TestCase):
    _DOM_STUB = _UIHarness._DOM_STUB
    _template_script = staticmethod(_UIHarness._template_script)
    _run_node = _UIHarness._run_node

    def test_disabled_queue_clear_uses_preview_confirm_and_keeps_switch_disabled(self):
        self._run_node(
            textwrap.dedent(r"""
          const api=globalThis.__MEDIAFLUX_STRM_TEST_API__;
          const base={running:false,enabled:false,progress:{},last_run:{},source_runtime:[]};
          api.renderStatus({...base,metadata_queue:{enabled:false,queued:13581,retry_wait:0,running:0,pending:13581}});
          const button=node('cancelStrmMetadataBacklogBtn');
          if(button.disabled)throw new Error('disabled sync hid queue cancellation');
          if(!node('strmMetadataQueueSummary').textContent.includes('13,581'))throw new Error('missing bounded count');
          node('strmMetadataEnabled').checked=false;
          const calls=[];
          let resolveConfirm;
          globalThis.appConfirm=options=>new Promise(resolve=>{resolveConfirm=resolve; if(!options.message.includes('不删除任何已落盘文件'))throw new Error('missing file safety copy');});
          globalThis.fetch=async(url,options)=>{
            calls.push({url:String(url),body:options?.body});
            if(String(url).endsWith('/preview'))return {ok:true,json:async()=>({preview:{count:13581},confirmation_token:'frozen'})};
            if(String(url).endsWith('/cancel-pending'))return {ok:true,json:async()=>({cancelled:13581,enabled:false})};
            return {ok:true,json:async()=>({...base,metadata_queue:{enabled:false,queued:0,retry_wait:0,running:0,pending:0}})};
          };
          (async()=>{
            const action=button.fire('click');
            for(let i=0;i<6;i++)await Promise.resolve();
            if(calls.length!==1)throw new Error('write before user confirmation');
            if(!button.disabled||button.getAttribute('aria-busy')!=='true')throw new Error('missing stable busy state');
            resolveConfirm(true);
            await action;
            if(calls.length!==3)throw new Error('expected preview + cancel + background status');
            if(JSON.parse(calls[1].body).confirmation_token!=='frozen')throw new Error('preview was not frozen');
            if(node('strmMetadataEnabled').checked)throw new Error('sync was enabled');
            if(!button.disabled||button.getAttribute('aria-busy')!=='false')throw new Error('empty queue control not settled');
          })().catch(error=>{console.error(error);process.exitCode=1;});
        """)
        )

    def test_zero_running_is_waiting_not_processing_and_running_drains_after_disable(
        self,
    ):
        self._run_node(
            textwrap.dedent(r"""
          const api=globalThis.__MEDIAFLUX_STRM_TEST_API__;
          const base={running:false,enabled:false,progress:{},last_run:{},source_runtime:[]};
          api.renderStatus({...base,metadata_queue:{enabled:true,queued:5,running:0,pending:5,consumer_active:true}});
          if(!node('strmProgressStage').textContent.includes('等待后台领取'))throw new Error('queued was misreported as processing');
          api.renderStatus({...base,metadata_queue:{enabled:false,queued:5,running:1,pending:6,consumer_active:true}});
          if(!node('strmProgressStage').textContent.includes('当前任务收尾中'))throw new Error('in-flight work was hidden');
        """)
        )

    def test_user_cancel_does_not_send_mutation(self):
        self._run_node(
            textwrap.dedent(r"""
          const api=globalThis.__MEDIAFLUX_STRM_TEST_API__;
          const base={running:false,enabled:false,progress:{},last_run:{},source_runtime:[],metadata_queue:{enabled:false,queued:1,pending:1}};
          api.renderStatus(base);
          const calls=[];
          globalThis.appConfirm=async()=>false;
          globalThis.fetch=async url=>{
            calls.push(String(url));
            return {ok:true,json:async()=>String(url).endsWith('/preview')?{preview:{count:1},confirmation_token:'frozen'}:base};
          };
          (async()=>{
            await node('cancelStrmMetadataBacklogBtn').fire('click');
            if(calls.some(url=>url.endsWith('/cancel-pending')))throw new Error('cancel sent mutation');
            if(node('cancelStrmMetadataBacklogBtn').disabled)throw new Error('cancel stranded control');
          })().catch(error=>{console.error(error);process.exitCode=1;});
        """)
        )
