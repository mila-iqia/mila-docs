## Office Hours
            
Come ask for help in {{ office_hours_data["location"] }}:
{% for date in office_hours_data["dates"] %}
* from {{ date.start }} to {{ date.end }} on {{ date.day }}s{% endfor %}