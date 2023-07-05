Environmental Impact
********************

These are our environmental impact calculations for the different clusters we have access to.

Mila Cluster
============

CO2 emissions for power consumption
-----------------------------------

The hardware for the Mila cluster is hosted in the province of Quebec,
where the electricity is produced by Hydro-Québec, almost exclusively
from hydroelectricity. The CO2 emissions are therefore very low,
and we can find the exact values in `CO₂ Emissions and Hydro-Québec Electricity,
1990-2021 <https://www.hydroquebec.com/data/developpement-durable/pdf/d-5647-affiche-co2-2021-an-vf.pdf>`_.

We use the most recent value in the table for the year 2021, which is 0.6 kg/MWh.
The Mila cluster consumes about 115 kW of power, and is running 24/7.

By multiplying power by time, we get that 150 kW * (24*365 hours) is equal to 1314000 kWh,
or 1314 MWh if we convert kilo- to mega-.

We multiply that by the CO2 emissions per MWh, and 1314 times 0.6 is 788.4 kg, which is less a ton of CO2.
Looking online, it is relatively easy to find Gold Standard carbon credits for $25 per ton of CO2,
so we can offset the entire Mila cluster by spending approximately $20 per year.

Hardware
--------

No estimate has been done about the environmental impact of manufacturing the hardware itself.

Digital Research Alliance of Canada Clusters
============================================

CO2 emissions for power consumption
-----------------------------------

Our current mega-allocation with DRAC is on the Narval, Beluga and Cedar clusters.

Narval and Beluga are hosted in the *École de technologie supérieure* in Montreal (QC)
so the same kind of reasoning and calculations apply as with the Mila cluster.

The Cedar cluster is hosted at the *Simon Fraser University* in Vancouver (BC),
in a province where 87% of the electricity is hydroelectricity.

However, we do not have access to the exact numbers for those clusters at the moment.

Hardware
--------

No estimate has been done about the environmental impact of manufacturing the hardware itself.
