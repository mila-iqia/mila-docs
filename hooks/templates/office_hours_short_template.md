Join us at the {% set count=0 -%}
{%- for date in office_hours_data["dates"] -%}
{%- if loop.index0 > 0 -%} / {%- endif -%}
 {{ date.day[:3] }} {{ date.start }}-{{ date.end }}
{%- set count = count+1 -%}
{%- endfor %} Office Hours in {{ office_hours_data["location"] }}