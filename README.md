# ADLAF_cancer-prognosis-prediction
---

## About this Project

Challenges such as insufficient connectivity of real biological networks and data noise still affect the accuracy of cancer prognosis prediction. To address these issues, this study proposes an adaptive dual-localized augmentation framework Adaptive Dual Local Augmentation(ADLA).
---
## Directory Layout

We assume the user is running the script on python version **3.9.19** and has set up a directory with the same structure.
~~~
    [your_directory]  
~~~
Python codes are in the subdirectory directory at **code**. 
~~~
    [your_directory]\\code  
~~~
all the **omics_data**  are in the subdirectory directory at **k_{k}\\data**. 
~~~
    [your_directory]\\k_{k}\\data\\raw  
~~~
the **relationships.xlsx**  is in the subdirectory directory at **data**. 
~~~
    [your_directory]\\data
~~~
all the **results**  are in the subdirectory directory at **k_{k}\\indicator**. 
~~~
    [your_directory]\\k_{k}\\indicator
~~~



<details>
<summary>\\k_{k}\\data</summary>

![image](https://github.com/user-attachments/assets/008e05d7-01d3-40bb-b5f0-169fefdea972)

</details> 
<details> 
<summary>Pretrain & Train</summary>
    
![image](https://github.com/user-attachments/assets/14d9ebc8-335b-4865-875d-97da71516b83)

</details>

<details> 
<summary>indicator</summary>
    
![image](https://github.com/user-attachments/assets/5b46721c-0ef9-496f-b4e3-4a3c635082a3)


</details>
    
---
## Notice

In addition to the code, we also provide the environment that the user needs to configure. You can choose the version that suits your operating environment,  the necessary python packages are in the requirement.txt .

---

## Step

1.prepare omics_dataCancertype.xlsx 

2.prepare the relationships.xlsx

3.run preADLAF.py

4.run trainADLAF.py

5.run CaseStudyADLAF.py 

---




