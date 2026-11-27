# Attribution

The seed subclasses `harbor.agents.terminus_2.terminus_2.Terminus2` from
[Harbor v0.26.0](https://github.com/harbor-framework/harbor/releases/tag/v0.26.0),
distributed under the Apache License 2.0. The license is included in
`LICENSE.harbor`.

The following prompt files are copied from the released Harbor 0.26.0 package.
Trailing whitespace in the timeout template is removed; the text is unchanged:

- `src/harbor/agents/terminus_2/templates/terminus-json-plain.txt` →
  `src/terminalbench_agent/prompts/system.txt`
- `src/harbor/agents/terminus_2/templates/timeout.txt` →
  `src/terminalbench_agent/prompts/timeout.txt`

The recipe adds an editable subclass, cheap-run defaults, and Beaker integration
code. Harbor's agent loop, environment implementation, and verifier remain
dependencies rather than copied source.

Task definitions are downloaded from
[TerminalBench 2.0](https://github.com/harbor-framework/terminal-bench-2/tree/69671fbaac6d67a7ef0dfec016cc38a64ef7a77c),
the commit in Harbor's registry. Task assets and oracle solutions are not
committed to this cookbook. Oracle solutions are not included in staged case
files.
