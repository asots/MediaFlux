"""执行仓库原始 GCID 重试函数：一次明确新操作与未知结果重发不能共用生命周期。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import TestCase


class GCIDRetryUIAuditTests(TestCase):
    @staticmethod
    def _run(outcomes):
        source = (
            Path(__file__).resolve().parents[1] / "app/templates/_gcid_scripts.html"
        ).read_text()
        start = source.index("    async function retryTask(")
        end = source.index("    byId('gcidImportRefresh')", start)
        function = source[start:end]
        script = """
const state={retryTokens:new Map()};let sequence=0;
const operationToken=()=>`token-${++sequence}`;
const calls=[];const busy=[];
const setBusy=(_button,value)=>busy.push(value);
const renderProgress=()=>{};const statusLabel=(value)=>value;
const loadHistory=async()=>{};
const outcomes=OUTCOMES;
const jsonRequest=async(_url,options)=>{
 calls.push(JSON.parse(options.body).operation_token);
 const outcome=outcomes.shift();
 if(outcome==='network-error')throw new Error('synthetic transport failure');
 return {task:{id:7,status:outcome,success_count:0,failed_count:1}};
};
FUNCTION
(async()=>{const count=outcomes.length;for(let i=0;i<count;i++)await retryTask({id:7},{});
console.log(JSON.stringify({calls,busy,pending:state.retryTokens.has(7)}));})().catch(error=>{console.error(error);process.exitCode=1;});
""".replace("OUTCOMES", json.dumps(outcomes)).replace("FUNCTION", function)
        result = subprocess.run(
            ["node", "-e", script], capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)

    def test_each_acknowledged_terminal_result_allows_a_new_explicit_retry(self):
        result = self._run(["failed", "partial_success", "success"])
        self.assertEqual(result["calls"], ["token-1", "token-2", "token-3"])
        self.assertFalse(result["pending"])
        self.assertEqual(result["busy"], [True, False] * 3)

    def test_unknown_transport_outcome_reuses_token_until_acknowledged(self):
        result = self._run(["network-error", "failed", "success"])
        self.assertEqual(result["calls"], ["token-1", "token-1", "token-2"])
        self.assertFalse(result["pending"])

    def test_running_receipt_does_not_release_token_before_terminal_ack(self):
        pending = self._run(["running", "network-error"])
        self.assertEqual(pending["calls"], ["token-1", "token-1"])
        self.assertTrue(pending["pending"])
        completed = self._run(["running", "failed", "success"])
        self.assertEqual(completed["calls"], ["token-1", "token-1", "token-2"])
        self.assertFalse(completed["pending"])
