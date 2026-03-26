# How to ask for help

We try to provide you different levels of help at Mila, for you to not stay stuck in your research.


``` mermaid
    graph LR
        User(😔):::emoji
        FAQ(<span>Check the FAQ</span>):::bigger
        Slack("`<span>Ask on Slack
        chan #mila-cluster</span>`"):::bigger
        Helpdesk(<span>Contact Mila helpdesk</span>):::bigger
        OfficeHours(<span>Ask your question to the Office Hours</span>):::bigger
        Yay(😃):::emoji

        User ==> FAQ
        FAQ ==> Slack
        Slack ==> Helpdesk
        Helpdesk ==> OfficeHours
        OfficeHours ==> Yay


        User@{ shape: text }
        Yay@{ shape: text }

        click FAQ "/help/faq/"
        click Slack "https://mila-umontreal.slack.com/archives/CFAS8455H"
        click Helpdesk "https://mila-iqia.atlassian.net/servicedesk/customer/portal/5"

        classDef bigger font-size:26px, fill:#4051b5, stroke:#4051b5, color:white
        classDef emoji font-size:40px, fill:transparent
```

## Check the FAQ
Your answers may be in the [FAQ page](../help/faq).

## Ask help on Slack
Feel free to ask your question on the [#mila-cluster](https://mila-umontreal.slack.com/archives/CFAS8455H) channel on the Mila Slack.

## Contact Mila support
Writing to the support is done through [this portal](https://mila-iqia.atlassian.net/servicedesk/customer/portal/5).

## Come to the Office Hours
Office Hours take place in Lab A:

* from 3pm to 5pm on Tuesdays
* from 2pm to 4pm on Wednesdays.

They also take place online at the same dates: the virtual invite is on your Mila calendar.