# Node profile description


<div style="box-sizing:unset">
<table>
   <tr>
      <th rowspan="2"> Name </th>
      <th colspan="3"> GPU </th>
      <th rowspan="2" style="min-width:1rem"> CPUs </th>
      <th rowspan="2" style="min-width:1rem"> Sockets </th>
      <th rowspan="2" style="min-width:1rem"> Cores/Socket </th>
      <th rowspan="2" style="min-width:1rem"> Threads/Core </th>
      <th rowspan="2" style="min-width:1rem"> Memory (GB) </th>
      <th rowspan="2" style="min-width:1rem"> TmpDisk (TB) </th>
      <th rowspan="2" style="min-width:1rem"> Arch </th>
      <th style="min-width:1rem"> Slurm Features </th>
   </tr>
   <tr>
      <th style="min-width:1rem"> Model </th>
      <th style="min-width:1rem"> Mem </th>
      <th style="min-width:1rem"> # </th>
      <th> GPU Arch and Memory </th>
   </tr>
   <tr>
      <td colspan="12"><h5 style="margin: 5px 0 0 0;">GPU Compute Nodes</h5></td>
   </tr>
   <tr>
      <td><b> cn-a[001-011] </b></td>
      <td> RTX8000 </td>
      <td>  48 </td>
      <td> 8 </td>
      <td> 40 </td>
      <td> 2 </td>
      <td> 20 </td>
      <td> 1 </td>
      <td> 384 </td>
      <td> 3.6 </td>
      <td> x86_64 </td>
      <td> turing,48gb </td>
   </tr>

   <tr>
      <td><b>cn-b[001-005]</b></td>
      <td> V100 </td>
      <td> 32 </td>
      <td> 8 </td>
      <td> 40 </td>
      <td> 2 </td>
      <td> 20 </td>
      <td> 1 </td>
      <td> 384 </td>
      <td> 3.6 </td>
      <td> x86_64 </td>
      <td> volta,nvlink,32gb </td>
   </tr>

   <tr>
      <td><b> cn-c[001-040] </b></td>
      <td> RTX8000 </td>
      <td> 48 </td>
      <td> 8 </td>
      <td> 64 </td>
      <td> 2 </td>
      <td> 32 </td>
      <td> 1 </td>
      <td> 384 </td>
      <td> 3 </td>
      <td> x86_64 </td>
      <td> turing,48gb </td>
   </tr>

   <tr>
      <td><b> cn-g[001-029] </b></td>
      <td> A100 </td>
      <td> 80 </td>
      <td> 4 </td>
      <td> 64 </td>
      <td> 2 </td>
      <td> 32 </td>
      <td> 1 </td>
      <td> 1024 </td>
      <td> 7 </td>
      <td> x86_64 </td>
      <td> ampere,nvlink,80gb </td>
   </tr>

   <tr>
      <td><b> cn-i001 </b></td>
      <td> A100 </td>
      <td> 80 </td>
      <td> 4 </td>
      <td> 64 </td>
      <td> 2 </td>
      <td> 32 </td>
      <td> 1 </td>
      <td> 1024 </td>
      <td> 3.6 </td>
      <td> x86_64 </td>
      <td> ampere,80gb </td>
   </tr>

   <tr>
      <td><b> cn-j001 </b></td>
      <td> A6000 </td>
      <td> 48 </td>
      <td> 8 </td>
      <td> 64 </td>
      <td> 2 </td>
      <td> 32 </td>
      <td> 1 </td>
      <td> 1024 </td>
      <td> 3.6 </td>
      <td> x86_64 </td>
      <td> ampere,48gb </td>
   </tr>

   <tr>
      <td><b> cn-k[001-004] </b></td>
      <td> A100 </td>
      <td> 40 </td>
      <td> 4 </td>
      <td> 48 </td>
      <td> 2 </td>
      <td> 24 </td>
      <td> 1 </td>
      <td> 512 </td>
      <td> 3.6 </td>
      <td> x86_64 </td>
      <td> ampere,nvlink,40gb </td>
   </tr>

   <tr>
      <td><b> cn-l[001-091] </b></td>
      <td> L40S </td>
      <td> 48 </td>
      <td> 4 </td>
      <td> 48 </td>
      <td> 2 </td>
      <td> 24 </td>
      <td> 1 </td>
      <td> 1024 </td>
      <td> 7 </td>
      <td> x86_64 </td>
      <td> lovelace,48gb </td>
   </tr>

   <tr>
      <td><b> cn-n[001-002] </b></td>
      <td> H100 </td>
      <td> 80 </td>
      <td> 8 </td>
      <td> 192 </td>
      <td> 2 </td>
      <td> 96 </td>
      <td> 1 </td>
      <td> 2048 </td>
      <td> 35 </td>
      <td> x86_64 </td>
      <td> hopper,nvlink,80gb </td>
   </tr>

   <tr>
      <td colspan="12"><h5 style="margin: 5px 0 0 0;">DGX Systems</h5></td>
   </tr>

   <tr>
      <td><b> cn-d[001-002] </b></td>
      <td> A100 </td>
      <td> 40 </td>
      <td> 8 </td>
      <td> 128 </td>
      <td> 2 </td>
      <td> 64 </td>
      <td> 1 </td>
      <td> 1024 </td>
      <td> 14 </td>
      <td> x86_64 </td>
      <td> ampere,nvlink,dgx,40gb </td>
   </tr>

   <tr>
      <td><b> cn-d[003-004] </b></td>
      <td> A100 </td>
      <td> 80 </td>
      <td> 8 </td>
      <td> 128 </td>
      <td> 2 </td>
      <td> 64 </td>
      <td> 1 </td>
      <td> 2048 </td>
      <td> 28 </td>
      <td> x86_64 </td>
      <td> ampere,nvlink,dgx,80gb </td>
   </tr>

   <tr>
      <td><b> cn-e[002-003] </b></td>
      <td> V100 </td>
      <td> 32 </td>
      <td> 8 </td>
      <td> 40 </td>
      <td> 2 </td>
      <td> 20 </td>
      <td> 1 </td>
      <td> 512 </td>
      <td> 7 </td>
      <td> x86_64 </td>
      <td> volta,nvlink,dgx,32gb </td>
   </tr>

   <tr>
      <td colspan="12"><h5 style="margin: 5px 0 0 0;">CPU Compute Nodes</h5></td>
   </tr>

   <tr>
      <td><b> cn-f[001-004] </b></td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> 32 </td>
      <td> 1 </td>
      <td> 32 </td>
      <td> 1 </td>
      <td> 256 </td>
      <td> 10 </td>
      <td> x86_64 </td>
      <td> rome </td>
   </tr>

   <tr>
      <td><b> cn-h[001-004] </b></td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> 64 </td>
      <td> 2 </td>
      <td> 32 </td>
      <td> 1 </td>
      <td> 768 </td>
      <td> 7 </td>
      <td> x86_64 </td>
      <td> milan </td>
   </tr>

   <tr>
      <td><b> cn-m[001-004] </b></td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> 96 </td>
      <td> 2 </td>
      <td> 48 </td>
      <td> 1 </td>
      <td> 1024 </td>
      <td> 7 </td>
      <td> x86_64 </td>
      <td> sapphire </td>     
   </tr>
</table>
</div>


## Special nodes and outliers

### DGX A100

DGX A100 nodes are NVIDIA appliances with 8 NVIDIA A100 Tensor Core GPUs. Each
GPU has either 40 GB or 80 GB of memory, for a total of 320 GB or 640 GB per
appliance. The GPUs are interconnected via 6 NVSwitches which allow for 600
GB/s point-to-point bandwidth (unidirectional) and a full bisection bandwidth
of 4.8 TB/s (bidirectional). See the table above for the specifications of each
appliance.

In order to run jobs on a DGX A100 with 40GB GPUs, add the flags below to your
Slurm commands::

    --gres=gpu:a100:<number> --constraint="dgx&ampere"

In order to run jobs on a DGX A100 with 80GB GPUs, add the flags below to your
Slurm commands::

    --gres=gpu:a100l:<number> --constraint="dgx&ampere"