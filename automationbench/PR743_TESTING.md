# Test the harness experiment review changes

This branch starts from `beaker/20260901-2200-automationbench-skills`
(`b64e1fc31a2d06a0ee26c9a395ca8535ac3ca11d`) and pins the local Beaker SDK
to Beaker AI PR #743 at `8891b5c536ad78548117f8188b54030a546ac04d`.

For hosted testing, select `automationbench-harness-pr743` when launching
AutomationBench Skills through the PR #743 preview. The new cookbook commit
requires a fresh spec build rather than reusing the earlier branch's image.

Hosted builds install the SDK and runtime wheels supplied by the deployed
build worker. They do not use the cookbook's SDK pin to select the runtime.
Launching this branch through another deployment does not test PR #743.

Before comparing optimization results, verify that the new run records:

- Proposal process version 7 in `controller-artifacts/skill.json`.
- Support for `history compare --review`.
- Default stopping patience 5, unless explicitly overridden.

The dependency pin and fresh image do not regenerate the task playbook by
themselves. The controller-owned proposal process is assembled from the
runtime in the new image.
