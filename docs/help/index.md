# How to get help

Need a hand getting your experiments running on the cluster? You'll find everything you need on this website. We also provide different levels of support to ensure you never stay stuck in your research. Our staff is dedicated to helping you overcome any technical hurdle, whether it's via email, on Slack, or in person.


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


-   :octicons-discussion-closed-24:{ .lg .middle } __Check the FAQ__

    ---

    Your answers may be in the **FAQ page**

    [:octicons-arrow-right-24: Check the FAQ](../help/faq)
    
-   :material-robot:{ .lg .middle } __Ask an AI agent__

    ---

    Maybe an AI agent can help you.

    You can click on the "Ask AI" button in the bottom-right corner


-   :fontawesome-brands-slack:{ .lg .middle } __Ask help on Slack__

    ---

    Feel free to ask your question on the **#mila-cluster** channel on the Mila Slack

    [:octicons-arrow-right-24: Ask your questions on Slack](https://mila-umontreal.slack.com/archives/CFAS8455H)


-   :octicons-code-of-conduct-24:{ .lg .middle } __Contact Mila support__

    ---

    Writing to the support is done through this portal

    [:octicons-arrow-right-24: Contact the Helpdesk](https://mila-iqia.atlassian.net/servicedesk/customer/portal/5)

-   :material-door-open:{ .lg .middle } __Come to the Office Hours__

    ---

    Office Hours take place in Lab A:

    * from 3pm to 5pm on Tuesdays
    * from 2pm to 4pm on Wednesdays.

    They also take place online at the same dates: the virtual invite is on your Mila calendar.


</div>
## 1. Try these resources first

Before reaching out, there's a 90% chance you'll find an appropriate and immediate solution by consulting the resources built specifically for the Mila community:

* **The Documentation & FAQ:** Most common issues regarding the cluster and environment setup are covered right here.
* **On-site Chatbot:** Our embedded specialized chatbot allows you to get quick answers or easily navigate our technical docs.
* **AI Assistants:** Feel free to leverage Mila-specific context and feed your own AI agent for helping your research. Whether you are using it for experimenting or troubleshooting, it will always be ready, directly in your environment.

## 2. Contact the right team

If you've checked the docs and still need assistance, we've got your back! To get the fastest resolution, please direct your request to the team that matches your needs:

| Team | Scope of Support | Contact Method |
|------|-----------------|----------------|
| MyMila | Community Integration. General administrative onboarding and community-related inquiries. | MyMila Portal |
| IT Support | Technical Onboarding & Access. Account creation, **SSH issues**, VPN, hardware, and access management. | Email or Slack |
| IDT | Scientific Computing. Experts for running jobs, software optimization, and cluster hurdles. | Slack or IDT Office Hours (Lab A) |

## 3. Reach out

Whatever the issue, we are here to help you get your experiments back on track. You can reach us through the following channels:

* **Email:** For technical and access issues, you can email the IT Support team directly.
* **Slack:** This is the fastest way to chat with us. Join the dedicated help channels to interact directly with the IT or IDT experts.
* **Office Hours:** Pay us a visit at Lab A or online for any questions/issues you may have when running your jobs on the clusters.

Let's get your research moving!
