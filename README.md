## CAPP 30254 Project
### Machine Learning For Public Policy

Our data comes from the Center for Medicare \& Medicaid Services 2008-2010 Data Entrepreneurs’ Synthetic Public Use File (DE-SynPUF).
The DE-SynPUF was created with the goal of providing a realistic set of claims data in the public domain while providing protection to the Medicare beneficiaries’ protected health information.
The data structure of the Medicare DE-SynPUF is very similar to the CMS Limited Data Sets, but with a smaller number of variables.
The DE-SynPUF also provides a robust set of metadata on the CMS claims data that have not been previously available in the public domain. 
Although the DE-SynPUF has very limited inferential research value to draw conclusions about Medicare beneficiaries due to the synthetic processes used to create the file, 
the Medicare DE-SynPUF does increase access to a realistic Medicare claims data file in a timely and less expensive manner to spur the 
innovation necessary to achieve the goals of better care for beneficiaries and improve the health of the population.


#### Codebook

The Codebook is a human-readable, read-only version of the project's data and serves as a quick reference for viewing the attributes of any given field in the project without having to download and interpret the data.

- reimb2010: Medicare reimbursements (USD) in the year 2010
- reimb2008, reimb2009: A patient's last 2 years of Medicare reimbursements (USD)
- age2010: The patient's age at the beginning of 2010
- male: Gender (binary)
- race: Race (categorical)
- heart.failure, kidney, cancer, copd, depression, diabetes, ihd, osteoporosis, arthritis, stroke: Indicator for whether a patient has any of these chronic conditions (as determined by whether a diagnosis code for one of the conditions appeared on one or more of their claims from 2008-2009)
- InpatientClaims: The number of inpatient claims a paitent had in 2008-2009 (episodes of care at a hospital)
- OutpatientClaims: The number of outpatient claims a patient had in 2008-2009 (episodes of care at institutional outpatient providers, e.g. hospital outpatient departments, rural health clinics, renal dialysis facilities, outpatient rehabilitation facilities, comprehensive outpatient rehabilitation facilities, and community mental health centers)
- OfficeVisit, EyeExam, EKG, xray, CTScan, PhysicalTherapy, Ambulance: Number of 2008-2009 claims a patient had for these big-ticket health costs
- acuity: Proportion of patient’s 2008-2009 costs that occurred in the most expensive month during that period
- costTrend: The correlation of the patient’s 2008-2009 monthly costs with the month number (1-24)
- monthsWithClaims: The number of months the patient incurrent healthcare costs from 2008-2009
