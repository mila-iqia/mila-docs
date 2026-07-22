---
title: How-tos and guides
description: Guides related to the complete lifecycle of a research experiment on the cluster — from first login to sharing results.
hide:
  - toc
---
# How-tos and guides

Eight steps cover the full lifecycle of a research experiment on the cluster. <!-- Guides are proposed for each one of these steps. -->

!!! tip "Getting started"
    If you want to learns the basics and have a step-by-step understanding on how to run a job on the cluster, the [Getting started](../getting_started/) section is for you!


<div class="wf2-portal">
  <div class="wf2-portal-header">
    <p class="wf2-eyebrow">Research workflow</p>
    <p class="wf2-lead">Click any step to open its guide</p>
  </div>

  <div class="wf2-board">


    <!-- Phase 1: Prepare -->
    <div class="wf2-lane wf2-lane-prepare">
      <div class="wf2-lane-header">
        <span class="wf2-lane-tag">Phase 1</span>
        <span class="wf2-lane-name">Prepare</span>
      </div>

      <div class="wf2-lane-steps">
        <a class="wf2-step wf2-s1" href="../userguides/login/">
          <!--<span class="wf2-num">01</span>-->
          <span class="wf2-icon">
            <svg xmlns="http://www.w3.org/2000/svg" height="26" width="26" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M451.5 160C434.9 160 418.8 164.5 404.7 172.7C388.9 156.7 370.5 143.3 350.2 133.2C378.4 109.2 414.3 96 451.5 96C537.9 96 608 166 608 252.5C608 294 591.5 333.8 562.2 363.1L491.1 434.2C461.8 463.5 422 480 380.5 480C294.1 480 224 410 224 323.5C224 322 224 320.5 224.1 319C224.6 301.3 239.3 287.4 257 287.9C274.7 288.4 288.6 303.1 288.1 320.8C288.1 321.7 288.1 322.6 288.1 323.4C288.1 374.5 329.5 415.9 380.6 415.9C405.1 415.9 428.6 406.2 446 388.8L517.1 317.7C534.4 300.4 544.2 276.8 544.2 252.3C544.2 201.2 502.8 159.8 451.7 159.8zM307.2 237.3C305.3 236.5 303.4 235.4 301.7 234.2C289.1 227.7 274.7 224 259.6 224C235.1 224 211.6 233.7 194.2 251.1L123.1 322.2C105.8 339.5 96 363.1 96 387.6C96 438.7 137.4 480.1 188.5 480.1C205 480.1 221.1 475.7 235.2 467.5C251 483.5 269.4 496.9 289.8 507C261.6 530.9 225.8 544.2 188.5 544.2C102.1 544.2 32 474.2 32 387.7C32 346.2 48.5 306.4 77.8 277.1L148.9 206C178.2 176.7 218 160.2 259.5 160.2C346.1 160.2 416 230.8 416 317.1C416 318.4 416 319.7 416 321C415.6 338.7 400.9 352.6 383.2 352.2C365.5 351.8 351.6 337.1 352 319.4C352 318.6 352 317.9 352 317.1C352 283.4 334 253.8 307.2 237.5z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Connect</span>
            <span class="wf2-desc">SSH &amp; auth</span>
          </div>

          <span class="wf2-chevron"></span>
        </a>
        <a class="wf2-step wf2-s1" href="../userguides/python_uv/">
          <!--<span class="wf2-num">02</span>-->
          <span class="wf2-icon">
            <svg xmlns="http://www.w3.org/2000/svg" height="26" width="26" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M259.1 73.5C262.1 58.7 275.2 48 290.4 48L350.2 48C365.4 48 378.5 58.7 381.5 73.5L396 143.5C410.1 149.5 423.3 157.2 435.3 166.3L503.1 143.8C517.5 139 533.3 145 540.9 158.2L570.8 210C578.4 223.2 575.7 239.8 564.3 249.9L511 297.3C511.9 304.7 512.3 312.3 512.3 320C512.3 327.7 511.8 335.3 511 342.7L564.4 390.2C575.8 400.3 578.4 417 570.9 430.1L541 481.9C533.4 495 517.6 501.1 503.2 496.3L435.4 473.8C423.3 482.9 410.1 490.5 396.1 496.6L381.7 566.5C378.6 581.4 365.5 592 350.4 592L290.6 592C275.4 592 262.3 581.3 259.3 566.5L244.9 496.6C230.8 490.6 217.7 482.9 205.6 473.8L137.5 496.3C123.1 501.1 107.3 495.1 99.7 481.9L69.8 430.1C62.2 416.9 64.9 400.3 76.3 390.2L129.7 342.7C128.8 335.3 128.4 327.7 128.4 320C128.4 312.3 128.9 304.7 129.7 297.3L76.3 249.8C64.9 239.7 62.3 223 69.8 209.9L99.7 158.1C107.3 144.9 123.1 138.9 137.5 143.7L205.3 166.2C217.4 157.1 230.6 149.5 244.6 143.4L259.1 73.5zM320.3 400C364.5 399.8 400.2 363.9 400 319.7C399.8 275.5 363.9 239.8 319.7 240C275.5 240.2 239.8 276.1 240 320.3C240.2 364.5 276.1 400.2 320.3 400z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Encapsulate</span>
            <span class="wf2-desc">Portability & environment set up</span>
          </div>
          <span class="wf2-chevron"></span>
        </a>

        <a class="wf2-step wf2-s1" href="../userguides/sharing_data/">
          <!--<span class="wf2-num">03</span>-->
          <span class="wf2-icon">
            <svg xmlns="http://www.w3.org/2000/svg" height="26" width="26" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M544 269.8C529.2 279.6 512.2 287.5 494.5 293.8C447.5 310.6 385.8 320 320 320C254.2 320 192.4 310.5 145.5 293.8C127.9 287.5 110.8 279.6 96 269.8L96 352C96 396.2 196.3 432 320 432C443.7 432 544 396.2 544 352L544 269.8zM544 192L544 144C544 99.8 443.7 64 320 64C196.3 64 96 99.8 96 144L96 192C96 236.2 196.3 272 320 272C443.7 272 544 236.2 544 192zM494.5 453.8C447.6 470.5 385.9 480 320 480C254.1 480 192.4 470.5 145.5 453.8C127.9 447.5 110.8 439.6 96 429.8L96 496C96 540.2 196.3 576 320 576C443.7 576 544 540.2 544 496L544 429.8C529.2 439.6 512.2 447.5 494.5 453.8z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Manage data</span>
            <span class="wf2-desc">Datasets &amp; storage</span>
          </div>
          <span class="wf2-chevron"></span>
        </a>
      </div>
    </div>

    <div class="wf2-phase-sep" aria-hidden="true">
      <div class="wf2-sep-top"></div>
      <div class="wf2-sep-arrow"></div>
      <div class="wf2-sep-bot"></div>
    </div>


    <!-- Phase 2: Execute -->
    <div class="wf2-lane wf2-lane-execute">
      <div class="wf2-lane-header">
        <span class="wf2-lane-tag">Phase 2</span>
        <span class="wf2-lane-name">Execute</span>
      </div>

      <div class="wf2-lane-steps">
        <a class="wf2-step wf2-s2" href="../examples/">
          <!--<span class="wf2-num">04</span>-->
          <span class="wf2-icon">
            <svg xmlns="http://www.w3.org/2000/svg" height="26" width="26" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M73.4 182.6C60.9 170.1 60.9 149.8 73.4 137.3C85.9 124.8 106.2 124.8 118.7 137.3L278.7 297.3C291.2 309.8 291.2 330.1 278.7 342.6L118.7 502.6C106.2 515.1 85.9 515.1 73.4 502.6C60.9 490.1 60.9 469.8 73.4 457.3L210.7 320L73.4 182.6zM288 448L544 448C561.7 448 576 462.3 576 480C576 497.7 561.7 512 544 512L288 512C270.3 512 256 497.7 256 480C256 462.3 270.3 448 288 448z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Write code</span>
            <span class="wf2-desc">Develop &amp; debug</span>
          </div>
          <span class="wf2-chevron"></span>
        </a>

        <a class="wf2-step wf2-s2" href="../userguides/slurm_guide/">
          <!--<span class="wf2-num">05</span>-->
          <span class="wf2-icon">
            <svg xmlns="http://www.w3.org/2000/svg" height="26" width="26" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M192 384L88.5 384C63.6 384 48.3 356.9 61.1 335.5L114 247.3C122.7 232.8 138.3 224 155.2 224L250.2 224C326.3 95.1 439.8 88.6 515.7 99.7C528.5 101.6 538.5 111.6 540.3 124.3C551.4 200.2 544.9 313.7 416 389.8L416 484.8C416 501.7 407.2 517.3 392.7 526L304.5 578.9C283.2 591.7 256 576.3 256 551.5L256 448C256 412.7 227.3 384 192 384L191.9 384zM464 224C464 197.5 442.5 176 416 176C389.5 176 368 197.5 368 224C368 250.5 389.5 272 416 272C442.5 272 464 250.5 464 224z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Submit job</span>
            <span class="wf2-desc">Slurm &amp; resources</span>
          </div>
          <span class="wf2-chevron"></span>
        </a>
      </div>
    </div>

    <div class="wf2-phase-sep" aria-hidden="true">
      <div class="wf2-sep-top"></div>
      <div class="wf2-sep-arrow"></div>
      <div class="wf2-sep-bot"></div>
    </div>


    <!-- Phase 3: Iterate -->
    <div class="wf2-lane wf2-lane-iterate">
      <div class="wf2-lane-header">
        <span class="wf2-lane-tag">Phase 3</span>
        <span class="wf2-lane-name">Iterate</span>
      </div>
      <div class="wf2-lane-steps">
        <a class="wf2-step wf2-s3" href="../userguides/gpu_efficiency/">
          <!--<span class="wf2-num">06</span>-->
          <span class="wf2-icon">
            <svg xmlns="http://www.w3.org/2000/svg" height="26" width="26" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M128 128C128 110.3 113.7 96 96 96C78.3 96 64 110.3 64 128L64 464C64 508.2 99.8 544 144 544L544 544C561.7 544 576 529.7 576 512C576 494.3 561.7 480 544 480L144 480C135.2 480 128 472.8 128 464L128 128zM534.6 214.6C547.1 202.1 547.1 181.8 534.6 169.3C522.1 156.8 501.8 156.8 489.3 169.3L384 274.7L326.6 217.4C314.1 204.9 293.8 204.9 281.3 217.4L185.3 313.4C172.8 325.9 172.8 346.2 185.3 358.7C197.8 371.2 218.1 371.2 230.6 358.7L304 285.3L361.4 342.7C373.9 355.2 394.2 355.2 406.7 342.7L534.7 214.7z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Monitor</span>
            <span class="wf2-desc">Job status &amp; metrics</span>
          </div>
          <span class="wf2-chevron"></span>
        </a>


        <a class="wf2-step wf2-s3" href="../userguides/wandb/">
          <!--<span class="wf2-num">07</span>-->
          <span class="wf2-icon">
            <svg xmlns="http://www.w3.org/2000/svg" height="26" width="26" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M434.8 54.1C446.7 62.7 451.1 78.3 445.7 91.9L367.3 288L512 288C525.5 288 537.5 296.4 542.1 309.1C546.7 321.8 542.8 336 532.5 344.6L244.5 584.6C233.2 594 217.1 594.5 205.2 585.9C193.3 577.3 188.9 561.7 194.3 548.1L272.7 352L128 352C114.5 352 102.5 343.6 97.9 330.9C93.3 318.2 97.2 304 107.5 295.4L395.5 55.4C406.8 46 422.9 45.5 434.8 54.1z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Optimize</span>
            <span class="wf2-desc">GPU utilization</span>
          </div>
          <span class="wf2-chevron"></span>
        </a>


        <a class="wf2-step wf2-s3" href="../userguides/reproducibility">
          <!--<span class="wf2-num">08</span>-->
          <span class="wf2-icon">
            <svg xmlns="http://www.w3.org/2000/svg" height="26" width="26" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M342.6 73.4C330.1 60.9 309.8 60.9 297.3 73.4L169.3 201.4C156.8 213.9 156.8 234.2 169.3 246.7C181.8 259.2 202.1 259.2 214.6 246.7L288 173.3L288 384C288 401.7 302.3 416 320 416C337.7 416 352 401.7 352 384L352 173.3L425.4 246.7C437.9 259.2 458.2 259.2 470.7 246.7C483.2 234.2 483.2 213.9 470.7 201.4L342.7 73.4zM160 416C160 398.3 145.7 384 128 384C110.3 384 96 398.3 96 416L96 480C96 533 139 576 192 576L448 576C501 576 544 533 544 480L544 416C544 398.3 529.7 384 512 384C494.3 384 480 398.3 480 416L480 480C480 497.7 465.7 512 448 512L192 512C174.3 512 160 497.7 160 480L160 416z"/></svg>
          </span>
          <div class="wf2-text">
            <span class="wf2-title">Share Results</span>
            <span class="wf2-desc">Reproducibility</span>
          </div>
          <span class="wf2-chevron"></span>
        </a>
      </div>
    </div>

  </div>
</div>
