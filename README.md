# ADLA_cancer-prognosis-prediction
---

## About this Project

Challenges such as insufficient connectivity of real biological networks and data noise still affect the accuracy of cancer prognosis prediction. To address these issues, this study proposes an adaptive dual-localized augmentation framework Adaptive Dual Local Augmentation(ADLA).
---
## Directory Layout

We assume the user is running the script on python version **3.9.19** and has set up a directory with the same structure.
~~~
    [your_directory]  
~~~
Python codes are in the subdirectory directory at **k_{k}\\code**. 
~~~
    [your_directory]\\k_{k}\\code  
~~~
all the **datasets**  are in the subdirectory directory at **k_{k}\\data**. 
~~~
    [your_directory]\\k_{k}\\data  
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
<summary>cancer</summary>

![image](https://github.com/user-attachments/assets/702e5c9e-645c-4ce6-9024-5bcd6faf76b1)

</details> 
<details> 
<summary>Model</summary>
    
![image](https://github.com/user-attachments/assets/ea67f735-0fd5-4193-a4a0-952fe10662a4)

</details>

<details> 
<summary>IndicatorsResults</summary>
    
![image](https://github.com/user-attachments/assets/03cbbd02-df02-49fa-aba7-92975d00c188)

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




