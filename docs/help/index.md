# How to ask for help

We try to provide you different levels of help at Mila, for you to not stay stuck in your research.


``` mermaid
    graph LR
        User(😔):::emoji
        FAQ(<span>Check the FAQ</span>):::bigger
        AI(<span>Ask an AI agent</span>):::bigger
        Slack("`<span>Ask on Slack
        chan #mila-cluster</span>`"):::bigger
        Helpdesk(<span>Contact Mila helpdesk</span>):::bigger
        OfficeHours(<span>Ask your question to the Office Hours</span>):::bigger
        Yay(😃):::emoji

        User ==> FAQ
        FAQ ==> AI
        AI ==> Slack
        Slack ==> Helpdesk
        Helpdesk ==> OfficeHours
        OfficeHours ==> Yay


        User@{ shape: text }
        Yay@{ shape: text }

        click FAQ "../help/faq/"
        click Slack "https://mila-umontreal.slack.com/archives/CFAS8455H"
        click Helpdesk "https://mila-iqia.atlassian.net/servicedesk/customer/portal/5"

        classDef bigger font-size:26px, fill:#4051b5, stroke:#4051b5, color:white
        classDef emoji font-size:40px, fill:transparent
```



</br >


<div class="grid cards" markdown>


-   <a href="../help/faq" class="full-card">
    :octicons-discussion-closed-24:{ .lg .middle } __Check the FAQ__

    ---

    Your answers may be in the **FAQ page**

    </a>
    
-   <a href="#" class="full-card">
    :material-robot:{ .lg .middle } __Ask an AI agent__

    ---

    Maybe an AI agent can help you.

    You can click on the "Ask AI" button in the bottom-right corner

    </a>

-   <a href="https://mila-umontreal.slack.com/archives/CFAS8455H" class="full-card">
    :fontawesome-brands-slack:{ .lg .middle } __Ask help on Slack__

    ---

    Feel free to ask your question on the **#mila-cluster** channel on the Mila Slack

    </a>


-   <a href="https://mila-iqia.atlassian.net/servicedesk/customer/portal/5" class="full-card">
    :octicons-code-of-conduct-24:{ .lg .middle } __Contact Mila support__

    ---

    Writing to the support is done through this portal

    </a>

-   <a href="#" class="full-card">
    :material-door-open:{ .lg .middle } __Come to the Office Hours__

    ---

    Office Hours take place in Lab A:

    * from 3pm to 5pm on Tuesdays
    * from 2pm to 4pm on Wednesdays.

    They also take place online at the same dates: the virtual invite is on your Mila calendar.

    </a>

</div>