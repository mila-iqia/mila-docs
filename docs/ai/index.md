---
title: AI Agents
description: >-
  Guides for AI coding tools available to Mila researchers, including Claude
  Code and Mila-specific skills.
hide:
  - toc
---

# AI agents

This section describes what Mila provides to empower you in your research with AI assistant related tools.

## Guides

<div class="grid cards important" markdown>

-   [:material-run-fast:{ .lg .middle } __Getting started with the Mila local agent__](local_agent.md)
    { .card }

    ---
    Step-by-step set up of an OpenCode harness to use the Mila local agent.

</div>

<div class="grid cards" markdown>

-   [:material-robot:{ .lg .middle } __Use the Mila local agent__](local_agent.md)
    { .card }

    ---
    In-depth explanation of the local agent hosted on the Mila cluster.

-   [:material-toy-brick:{ .lg .middle } __Add Mila Skills__](mila_skills.md)
    { .card }

    ---
    Install the Mila skills to automate common workflows.

-   [:material-book:{ .lg .middle } __Learn from Mila shared knowledge__](add_context.md)
    { .card }

    ---
    Give your AI assistant access to the Mila documentation through the Mila Docs
    MCP server.

</div>

## Overview of an agent ecosystem

An AI agent combines several parts that work together: a **model** does the thinking, and a program called a
**harness** provides everything else needed to get work done — information
to read, tools to use, and rules to follow.

The model is a large language model (LLM): it reads text and writes text.
On its own, it cannot open files or run programs. The harness makes the
model useful by running a repeating cycle called the **agent loop**:

1. **Think** – The model reviews the request and everything it currently
   knows, then decides what to do next.
2. **Act** – The model asks the harness to run a tool, such as reading a
   file or launching a computation. The harness checks its permission
   rules first and requests approval for sensitive actions.
3. **Observe** – The result of the action is added to what the model
   knows, and the cycle repeats until the task is complete.

<div class="wf2-loop">
  <svg viewBox="0 0 680 290" role="img" aria-labelledby="wf2-loop-title" xmlns="http://www.w3.org/2000/svg">
    <title id="wf2-loop-title">The agent loop: the researcher's goal goes to the model, which thinks; the harness checks permissions and runs the tool, then adds the result to the context; the cycle repeats until the model decides the task is complete and reports to the researcher.</title>
    <defs>
      <marker id="wf2-loop-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path class="lp-head" d="M0 0 L10 5 L0 10 z"/>
      </marker>
    </defs>
    <!-- Entry and exit -->
    <rect class="lp-io" x="10" y="42" width="130" height="36" rx="18"/>
    <text class="lp-io-text" x="75" y="64.5" text-anchor="middle">Researcher goal</text>
    <path class="lp-edge" d="M140 60 H251" marker-end="url(#wf2-loop-arrow)"/>
    <rect class="lp-io" x="528" y="42" width="146" height="36" rx="18"/>
    <text class="lp-io-text" x="601" y="64.5" text-anchor="middle">Report to researcher</text>
    <path class="lp-edge" d="M425 60 H524" marker-end="url(#wf2-loop-arrow)"/>
    <text class="lp-label" x="475" y="51" text-anchor="middle">task complete</text>
    <!-- Cycle edges -->
    <path class="lp-edge lp-loop" d="M398 104 Q475 118 485 186" marker-end="url(#wf2-loop-arrow)"/>
    <text class="lp-label" x="486" y="136">tool call</text>
    <path class="lp-edge lp-loop" d="M400 230 H284" marker-end="url(#wf2-loop-arrow)"/>
    <text class="lp-label" x="342" y="222" text-anchor="middle">result</text>
    <path class="lp-edge lp-loop" d="M195 186 Q205 118 282 104" marker-end="url(#wf2-loop-arrow)"/>
    <text class="lp-label" x="190" y="136" text-anchor="end">updated context</text>
    <text class="lp-center" x="340" y="155" text-anchor="middle">↻ repeats until the task is complete</text>
    <!-- Think (model) -->
    <rect class="lp-node lp-model" x="255" y="20" width="170" height="80" rx="10"/>
    <text class="lp-title" x="271" y="46">Think</text>
    <rect class="lp-badge-model" x="357" y="32" width="54" height="18" rx="9"/>
    <text class="lp-badge-text lp-badge-model-text" x="384" y="45" text-anchor="middle">Model</text>
    <text class="lp-sub" x="271" y="70">Reviews the context,</text>
    <text class="lp-sub" x="271" y="87">decides the next step</text>
    <!-- Act (harness) -->
    <rect class="lp-node lp-harness" x="400" y="190" width="170" height="80" rx="10"/>
    <text class="lp-title" x="416" y="216">Act</text>
    <rect class="lp-badge-harness" x="492" y="202" width="64" height="18" rx="9"/>
    <text class="lp-badge-text lp-badge-harness-text" x="524" y="215" text-anchor="middle">Harness</text>
    <text class="lp-sub" x="416" y="240">Checks permissions,</text>
    <text class="lp-sub" x="416" y="257">runs the tool</text>
    <!-- Observe (harness) -->
    <rect class="lp-node lp-harness" x="110" y="190" width="170" height="80" rx="10"/>
    <text class="lp-title" x="126" y="216">Observe</text>
    <rect class="lp-badge-harness" x="202" y="202" width="64" height="18" rx="9"/>
    <text class="lp-badge-text lp-badge-harness-text" x="234" y="215" text-anchor="middle">Harness</text>
    <text class="lp-sub" x="126" y="240">Adds the result</text>
    <text class="lp-sub" x="126" y="257">to the context</text>
  </svg>
</div>

What the model knows at any moment — instructions, project files, notes
from earlier sessions, step-by-step guides — is called its **context**.
What it can do — run commands, edit code, reach external services —
depends on the **tools** the harness provides. The diagram below shows
these parts and how they connect, from the researcher who states the goal
to the external services the agent can reach.

<div class="wf2-portal">
  <!--<div class="wf2-portal-header">
    <p class="wf2-eyebrow">Overview of an agent ecosystem</p>
  </div>-->

  <div class="wf2-stack">

    <div class="wf2-lane wf2-lane-model wf2-band">
      <div class="wf2-lane-header">
        <span class="wf2-lane-name">Researcher</span>
      </div>

      <!--
      <div class="wf2-lane-steps">
        <div class="wf2-step wf2-s0 wf2-static" title="Gives the task in natural language and answers the agent's questions. The agent acts on behalf of the researcher.">
          <span class="wf2-icon">
            <svg height="26" width="26" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M9 5a4 4 0 0 1 4 4 4 4 0 0 1-4 4 4 4 0 0 1-4-4 4 4 0 0 1 4-4m0 10c2.67 0 8 1.34 8 4v2H1v-2c0-2.66 5.33-4 8-4m7.76-9.64c2.02 2.2 2.02 5.25 0 7.27l-1.68-1.69c.84-1.18.84-2.71 0-3.89zM20.07 2c3.93 4.05 3.9 10.11 0 14l-1.63-1.63c2.77-3.18 2.77-7.72 0-10.74z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Request</span>
            <span class="wf2-desc">Describes the task</span>
          </div>
        </div>
        <div class="wf2-step wf2-s0 wf2-static" title="Accepts or refuses actions that the permission rules flag, such as deleting files or sending data outside.">
          <span class="wf2-icon">
            <svg height="26" width="26" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="m23 12-2.44-2.78.34-3.68-3.61-.82-1.89-3.18L12 3 8.6 1.54 6.71 4.72l-3.61.81.34 3.68L1 12l2.44 2.78-.34 3.69 3.61.82 1.89 3.18L12 21l3.4 1.46 1.89-3.18 3.61-.82-.34-3.68zm-13 5-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Approve</span>
            <span class="wf2-desc">Validates sensitive actions</span>
          </div>
        </div>
        <div class="wf2-step wf2-s0 wf2-static" title="Redirects the agent when its answer or its plan is wrong.">
          <span class="wf2-icon">
            <svg height="26" width="26" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M18 8H6V6h12zm0 3H6V9h12zm0 3H6v-2h12zm4-10a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h14l4 4z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Correct</span>
            <span class="wf2-desc">Gives feedback</span>
          </div>
        </div>
      </div>
      -->
    </div>

    <div class="wf2-vsep" aria-hidden="true"></div>

  <div class="wf2-board">

    <div class="wf2-lane wf2-lane-prepare">
      <div class="wf2-lane-header">
        <span class="wf2-lane-tag">Context</span>
        <span class="wf2-lane-name">What is known</span>
      </div>
      <div class="wf2-lane-steps">
        <ul class="wf2-list wf2-s1">
          <li><b>System prompt</b><span class="details"> – Role, rules, style</span></li>
          <li><b>AGENTS.md</b><span class="details"> – Project instructions</span></li>
          <li><b>Memory</b><span class="details"> – Facts kept across sessions</span></li>
          <li><b><a href="context/">Context description</a></b><span class="details"> – Context-specific information</span></li>
          <li><b><a href="mila_skills/">Skills</a></b><span class="details"> – Know-how loaded on demand</span></li>
        </ul>
      </div>
    </div>

    <div class="wf2-phase-sep" aria-hidden="true">
      <div class="wf2-sep-top"></div>
      <div class="wf2-sep-arrow"></div>
      <div class="wf2-sep-bot"></div>
    </div>

    <div class="wf2-lane wf2-lane-execute">
      <div class="wf2-lane-header">
        <span class="wf2-lane-tag">Harness</span>
        <span class="wf2-lane-name">How all works together</span>
      </div>

      <div class="wf2-lane-steps">
        <ul class="wf2-list wf2-s2">
          <li><b><a href="local_agent/">Model (LLM)</a></b><span class="details"> – Reasons, decides, writes</span></li>
          <li><b>Agent loop</b><span class="details"> – Think, act, observe</span></li>
          <li><b>Context window</b><span class="details"> – What the model sees</span></li>
          <li><b>Permissions</b><span class="details"> – What is allowed</span></li>
          <li><b>Hooks</b><span class="details"> – Event-triggered scripts</span></li>
        </ul>
        <!--
        <a class="wf2-step wf2-s2" href="claude_code/" title="Install Claude Code and extend it with skills from a marketplace.">
          <span class="wf2-icon">
            <svg height="26" width="26" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18a2.5 2.5 0 0 0 2.5-2.5A2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Set up Claude Code</span>
            <span class="wf2-desc">Install the harness</span>
          </div>
          <span class="wf2-chevron"></span>
        </a>
        -->
      </div>
    </div>

    <div class="wf2-phase-sep wf2-sep-both" aria-hidden="true">
      <div class="wf2-sep-top"></div>
      <div class="wf2-sep-arrow"></div>
      <div class="wf2-sep-bot"></div>
    </div>

    <div class="wf2-lane wf2-lane-iterate">
      <div class="wf2-lane-header">
        <span class="wf2-lane-tag">Actions</span>
        <span class="wf2-lane-name">What can be done</span>
      </div>
      <div class="wf2-lane-steps">
        <ul class="wf2-list wf2-s3">
          <li><b>Tool calls</b><span class="details"> – Model asks, harness runs</span></li>
          <li><b>Built-in tools</b><span class="details"> – Read, write, shell, web</span></li>
          <li><b>Slash commands</b><span class="details"> – Shortcuts typed by a human</span></li>
          <li><b><a href="mcp/">MCP servers</a></b><span class="details"> – Standard plug to services</span></li>
          <li><b>Plugins</b><span class="details"> – Bundles of skills, MCP, hooks</span></li>
        </ul>
      </div>
    </div>

  </div>

    <div class="wf2-board wf2-vrow" aria-hidden="true">
      <div class="wf2-vcell wf2-vcell-empty"></div>
      <div class="wf2-phase-sep" aria-hidden="true">
        <div class="wf2-sep-top"></div>
        <div class="wf2-sep-arrow"></div>
        <div class="wf2-sep-bot"></div>
      </div>
      <div class="wf2-vcell">Runs tools, launches sub-agents</div>
      <div class="wf2-phase-sep" aria-hidden="true">
        <div class="wf2-sep-top"></div>
        <div class="wf2-sep-arrow"></div>
        <div class="wf2-sep-bot"></div>
      </div>
      <div class="wf2-vcell">MCP servers, web tools</div>
    </div>

    <div class="wf2-lane wf2-lane-model wf2-band">
      <div class="wf2-lane-header">
      <span class="wf2-lane-name">Outside the harness</span>
      </div>
      <div class="wf2-lane-steps">
        <ul class="wf2-list">
          <li><b>Sub-agents</b><span class="details"> – Isolated context, summary back</span></li>
          <li><b>Execution environment</b><span class="details"> – Cluster, repository, terminal</span></li>
          <li><b>External services</b><span class="details"> – Reached through MCP servers</span></li>
        </ul>
      </div>
    </div>

  </div>

</div>
