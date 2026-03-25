# How to ask for help

We try to provide you different levels of help at Mila, for you to not stay stuck in your research.


``` mermaid
    %%{init: {'theme': 'base', 'themeVariables': {'fontColor': 'purple', 'fontSize': '20px', 'fontFamily': 'Inter'}}}%%
    graph LR
        User(😔):::emoji
        Slack("`Ask on Slack
        chan #mila-cluster`"):::bigger
        Yay(😃):::emoji

        User ==> FAQ(Check the FAQ):::bigger
        FAQ ==> Slack
        Slack ==> Helpdesk(Contact Mila helpdesk):::bigger
        Helpdesk ==> OfficeHours(Ask your question to the Office Hours):::bigger
        OfficeHours ==> Yay


        User@{ shape: text }
        Yay@{ shape: text }

        click FAQ "/help/faq/"
        click Helpdesk "https://mila-iqia.atlassian.net/servicedesk/customer/portal/5"

        classDef bigger font-size:26px
        classDef emoji font-size:40px
```

## Check the FAQ
Your answers may be in the [FAQ page](/help/faq).

## Ask help on Slack
Feel free to ask your question on the `#mila-cluster` channel on the Mila Slack.

## Contact Mila support
Writing to the support is done through [this portal](https://mila-iqia.atlassian.net/servicedesk/customer/portal/5).

## Come to the Office Hours
Office Hours take place in Lab A:

* from 3pm to 5pm on Tuesdays
* from 2pm to 4pm on Wednesdays.

They also take place online at the same dates: the virtual invite is on your Mila calendar.