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
Python codes are in the subdirectory directory at **code**. 
~~~
    [your_directory]/code  
~~~
all the **datasets**  are in the subdirectory directory at **cancer**. 
~~~
    [your_directory]/cancer  
~~~
all the **results**  are in the subdirectory directory at **IndicatorsResult**. 
~~~
    [your_directory]/IndicatorsResult 
~~~
all the **models**  are in the subdirectory directory at **Model**. 
~~~
    [your_directory]/Model
~~~

</details></summary>cancer</summary>
    ├── cancer
    │ 	 ├── BRCA           #cancer1
    │ 	 │ 	 ├── raw		  
    │ 	 │ 	 │	 ├── omicsdata_BRCA.csv  
    │ 	 │ 	 └── process
    │ 	 │ 	 	 └── # The corresponding file will be generated after running preADLA.py	    
    │ 	 └── CESC           #cancer2
    │ 	     ├── raw		  
    │ 	  	 │	 ├── omicsdata_BRCA.csv  
    │ 	  	 └── process
    │ 	  	 	 └── # The corresponding file will be generated after running preADLA.py			    

</details>
</details></summary>Model</summary>
    ├── Train
    │ 	├── BRCA #cancer
    │       ├── # The corresponding file will be generated after running preADLA.py		
    ├── Pretrain
    │   ├── BRCA #cancer
    │       ├── # The corresponding file will be generated after running preADLA.py		

</details>

---
## Notice

In addition to the code, we also provide the environment that the user needs to configure.You can choose the version that suits your operating environment,  the necessary python packages are in the requirement.txt .

---






