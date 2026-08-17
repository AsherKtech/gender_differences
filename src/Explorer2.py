# %% cell 1
import pandas as pd



# %% cell 2
# Load the dataset

# Read the csv file into a DataFrame.
df = pd.read_csv("/data/home/asher.katz/Projects/gender_differences/data/raw/ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.csv")

# Display the first few rows of the DataFrame.
print(df.head())

# Get information about the DataFrame.
print(df.info())
print(df.describe())


# %% cell 3
# Find columns which have all the same value (i.e., no variance).
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
print("Columns with no variance:", constant_columns)
# Find columns which are all Nan
all_nan_columns = [col for col in df.columns if df[col].isnull().all()]
print("Columns with all values as NaN:", all_nan_columns)
# what is the size of all_nan_columns
print("Size of columns with all values as NaN:", len(all_nan_columns))
#how many columns are there in total
total_columns = len(df.columns)
print("Total number of columns:", total_columns)


# %% cell 4
# get list of all of the column names in the dataset
column_names = df.columns.tolist()
for i in range(0, len(column_names)):
    print("'", column_names[i], "',", sep="")

# %% cell 4.5
# get list of all of the column names in the dataset
column_names = df.columns.tolist()
column_names

#%% cell 5
ess_column_descriptions = {
    # Metadata & Sample Weights
    "name": "Title of the ESS dataset",
    "essround": "ESS round/wave number",
    "edition": "Dataset edition version",
    "proddate": "Dataset production date",
    "idno": "Respondent's unique identification number",
    "cntry": "Country code (2-letter ISO code)",
    "dweight": "Design weight (corrects for selection probabilities)",
    "pspwght": "Post-stratification weight including design weight",
    "pweight": "Population size weight",
    "anweight": "Analysis weight (combines post-stratification and population weights)",
    "prob": "Sampling probability",
    "stratum": "Sampling stratum identifier",
    "psu": "Primary Sampling Unit identifier",

    # Media & Internet Usage
    "netuse": "Personal use of the internet",
    "netusoft": "How often the internet is used",
    "netustm": "Internet use duration on a typical day (minutes)",
    "nwspol": "Time spent watching/reading/listening to news about politics (minutes)",
    "nwsppol": "Time spent following news/politics (older round metric)",
    "nwsptot": "Total time spent following news/current affairs (older round metric)",
    "rdpol": "Radio listening to politics/current affairs",
    "rdtot": "Total radio listening time",
    "tvpol": "TV watching of news/politics",
    "tvtot": "Total TV watching time",

    # Social Trust & Morality
    "pplfair": "Most people try to take advantage of you, or try to be fair (0-10)",
    "pplhlp": "Most of the time people are helpful, or mostly looking out for themselves (0-10)",
    "ppltrst": "Most people can be trusted, or you can't be too careful (0-10)",

    # Political Engagement & Efficacy
    "polintr": "How interested in politics (1-4)",
    "actrolg": "Able to take active role in political group (1-5)",
    "actrolga": "Able to take active role in political group (revised scale)",
    "clsprty": "Feel closer to a particular political party than all other parties",
    "prtdgcl": "How close respondent feels to that preferred party",
    "lrscale": "Placement on political left-right scale (0-10)",
    "cptppol": "Confident in own ability to participate in politics",
    "cptppola": "Confident in own ability to participate in politics (revised scale)",
    "polcmpl": "Politics too complicated to understand",
    "poldcs": "How often system allows people to influence political decisions",
    "psppipl": "Political system allows people to have a say in government",
    "psppipla": "Political system allows people to have a say in government (revised scale)",
    "psppsgv": "Political system allows people to have influence on politics",
    "psppsgva": "Political system allows people to have influence on politics (revised scale)",

    # Political Actions & Participation (Last 12 Months)
    "contplt": "Contacted politician or government official",
    "wrkprty": "Worked in a political party or action group",
    "wrkorg": "Worked in another organization or association",
    "badge": "Worn or displayed a campaign badge/sticker",
    "sgnptit": "Signed a petition",
    "pbldmn": "Taken part in a lawful public demonstration",
    "bctprd": "Boycotted certain products",
    "pstplonl": "Posted or shared political content online",
    "mmbprty": "Member of political party",
    "ptcpplt": "Participated in political activities",
    "vote": "Voted in the last national election",

    # Trust in Institutions
    "trstprl": "Trust in national parliament (0-10)",
    "trstlgl": "Trust in the legal system (0-10)",
    "trstplc": "Trust in the police (0-10)",
    "trstplt": "Trust in politicians (0-10)",
    "trstprt": "Trust in political parties (0-10)",
    "trstep": "Trust in the European Parliament (0-10)",
    "trstun": "Trust in the United Nations (0-10)",

    # Satisfaction & System Attitudes
    "stflife": "How satisfied with life as a whole (0-10)",
    "stfeco": "How satisfied with present state of economy in country (0-10)",
    "stfgov": "How satisfied with national government (0-10)",
    "stfdem": "How satisfied with the way democracy works in country (0-10)",
    "stfedu": "State of education in country nowadays (0-10)",
    "stfhlth": "State of health services in country nowadays (0-10)",

    # Values, Social Attitudes & Equality
    "gincdif": "Government should reduce differences in income levels (1-5)",
    "freehms": "Gays and lesbians should be free to live life as they wish (1-5)",
    "hmsfmlsh": "Ashamed if a close family member were gay or lesbian",
    "hmsacld": "Gay and lesbian couples should have the right to adopt children",
    "euftf": "European unification: go further or gone too far (0-10)",
    "dmcntov": "Democracy in country: overall evaluation",
    "etapapl": "Equal treatment for all political opinions",
    "implvdm": "Importance of living in a democratically governed country",
    "prtyban": "Ban on political parties that wish to overthrow democracy",
    "scnsenv": "Science and technology make the environment better",

    # Immigration Attitudes
    "imsmetn": "Allow many/few immigrants of same race/ethnic group as majority",
    "imdfetn": "Allow many/few immigrants of different race/ethnic group from majority",
    "impcntr": "Allow many/few immigrants from poorer countries outside Europe",
    "imbgeco": "Immigration bad or good for country's economy (0-10)",
    "imueclt": "Country's cultural life undermined or enriched by immigrants (0-10)",
    "imwbcnt": "Immigrants make country a worse or better place to live (0-10)"
}

# --- Country-Specific Party Variables ---
# ESS uses country codes suffixed to prefixes for party choice/closeness/membership:
# prtv*: Party voted for in last national election (e.g., prtvede1 = Germany party vote)
# prtcl*: Party feel closer to (e.g., prtclbgb = UK party closeness)
# prtmb*: Party member of (e.g., prtmbafi = Finland party membership)

country_party_variables = [
    'prtvbde1', 'prtvcde1', 'prtvdde1', 'prtvede1', 'prtvbde2', 'prtvcde2', 'prtvdde2', 'prtvede2', 
    'prtvlt1', 'prtvalt1', 'prtvblt1', 'prtvlt2', 'prtvalt2', 'prtvblt2', 'prtvlt3', 'prtvalt3', 'prtvblt3', 
    'prtvtal', 'prtvtaat', 'prtvtbat', 'prtvtcat', 'prtvtabe', 'prtvtbbe', 'prtvtcbe', 'prtvtdbe', 'prtvtbg', 
    'prtvtabg', 'prtvtbbg', 'prtvtcbg', 'prtvtdbg', 'prtvtach', 'prtvtbch', 'prtvtcch', 'prtvtdch', 'prtvtech', 
    'prtvtfch', 'prtvtgch', 'prtvtcy', 'prtvtacy', 'prtvtbcy', 'prtvtacz', 'prtvtbcz', 'prtvtccz', 'prtvtdcz', 
    'prtvtecz', 'prtvtadk', 'prtvtbdk', 'prtvtcdk', 'prtvtddk', 'prtvtaee', 'prtvtbee', 'prtvtcee', 'prtvtdee', 
    'prtvteee', 'prtvtfee', 'prtvtgee', 'prtvtaes', 'prtvtbes', 'prtvtces', 'prtvtdes', 'prtvtees', 'prtvtfi', 
    'prtvtafi', 'prtvtbfi', 'prtvtcfi', 'prtvtdfi', 'prtvtafr', 'prtvtbfr', 'prtvtcfr', 'prtvtdfr', 'prtvtgb', 
    'prtvtagb', 'prtvtbgb', 'prtvtcgb', 'prtvtbgr', 'prtvtcgr', 'prtvthr', 'prtvtahr', 'prtvtahu', 'prtvtbhu', 
    'prtvtchu', 'prtvtdhu', 'prtvtehu', 'prtvtfhu', 'prtvtie', 'prtvtaie', 'prtvtbie', 'prtvtcie', 'prtvtail', 
    'prtvtbil', 'prtvtcil', 'prtvtais', 'prtvtbis', 'prtvtcis', 'prtvtbit', 'prtvtcit', 'prtvtlv', 'prtvtalv', 
    'prtvtbnl', 'prtvtcnl', 'prtvtdnl', 'prtvtenl', 'prtvtfnl', 'prtvtgnl', 'prtvtno', 'prtvtano', 'prtvtbno', 
    'prtvtapl', 'prtvtbpl', 'prtvtcpl', 'prtvtdpl', 'prtvtapt', 'prtvtbpt', 'prtvtcpt', 'prtvtro', 'prtvtaro', 
    'prtvtru', 'prtvtaru', 'prtvtbru', 'prtvtcru', 'prtvtdru', 'prtvtse', 'prtvtase', 'prtvtbse', 'prtvtcse', 
    'prtvtbsi', 'prtvtcsi', 'prtvtdsi', 'prtvtesi', 'prtvtfsi', 'prtvtask', 'prtvtbsk', 'prtvtcsk', 'prtvtdsk', 
    'prtvtatr', 'prtvtaua', 'prtvtbua', 'prtvtcua', 'prtvtxk', 'prtvtrs', 'prtvtme', 'prtclal', 'prtclaat', 
    'prtclcat', 'prtcldat', 'prtclabe', 'prtclbbe', 'prtclcbe', 'prtcldbe', 'prtclbg', 'prtclabg', 'prtclbbg', 
    'prtclcbg', 'prtcldbg', 'prtclach', 'prtclbch', 'prtclcch', 'prtcldch', 'prtclech', 'prtclfch', 'prtclgch', 
    'prtclask', 'prtclbsk', 'prtclcsk', 'prtcldsk', 'prtclcy', 'prtclacy', 'prtclbcy', 'prtclacz', 'prtclbcz', 
    'prtclccz', 'prtcldcz', 'prtclecz', 'prtclbde', 'prtclcde', 'prtcldde', 'prtclede', 'prtcladk', 'prtclbdk', 
    'prtclcdk', 'prtclddk', 'prtclaee', 'prtclbee', 'prtclcee', 'prtcldee', 'prtcleee', 'prtclfee', 'prtclgee', 
    'prtclaes', 'prtclbes', 'prtclces', 'prtcldes', 'prtclees', 'prtclfes', 'prtclfi', 'prtclafi', 'prtclbfi', 
    'prtclcfi', 'prtcldfi', 'prtclefi', 'prtclafr', 'prtclbfr', 'prtclcfr', 'prtcldfr', 'prtclefr', 'prtclffr', 
    'prtclgb', 'prtclagb', 'prtclbgb', 'prtclcgb', 'prtclbgr', 'prtclcgr', 'prtclhr', 'prtclahr', 'prtclahu', 
    'prtclbhu', 'prtclchu', 'prtcldhu', 'prtclehu', 'prtclfhu', 'prtclghu', 'prtclaie', 'prtclbie', 'prtclcie', 
    'prtcldie', 'prtcleie', 'prtclail', 'prtclbil', 'prtclcil', 'prtcldil', 'prtclais', 'prtclbis', 'prtclcis', 
    'prtclbit', 'prtclcit', 'prtcldit', 'prtcllt', 'prtclalt', 'prtclblt', 'prtcllv', 'prtclalv', 'prtclnl', 
    'prtclbnl', 'prtclcnl', 'prtcldnl', 'prtclenl', 'prtclfnl', 'prtclno', 'prtclano', 'prtclbno', 'prtclbpl', 
    'prtclcpl', 'prtcldpl', 'prtclepl', 'prtclfpl', 'prtclgpl', 'prtclhpl', 'prtclbpt', 'prtclcpt', 'prtcldpt', 
    'prtclept', 'prtclro', 'prtclaro', 'prtclru', 'prtclaru', 'prtclbru', 'prtclcru', 'prtcldru', 'prtclse', 
    'prtclase', 'prtclbse', 'prtclcse', 'prtclbsi', 'prtclcsi', 'prtcldsi', 'prtclesi', 'prtclfsi', 'prtclatr', 
    'prtclaua', 'prtclbua', 'prtclcua', 'prtcldua', 'prtclxk', 'prtclrs', 'prtclme', 'prtmbaat', 'prtmbabe', 
    'prtmbbbe', 'prtmbcbe', 'prtmbbg', 'prtmbabg', 'prtmbbbg', 'prtmbach', 'prtmbbch', 'prtmbcch', 'prtmbcy', 
    'prtmbacz', 'prtmbbcz', 'prtmbbde', 'prtmbcde', 'prtmbadk', 'prtmbbdk', 'prtmbaee', 'prtmbbee', 'prtmbcee', 
    'prtmbaes', 'prtmbbes', 'prtmbfi', 'prtmbafi', 'prtmbbfi', 'prtmbafr', 'prtmbbfr', 'prtmbcfr', 'prtmbgb', 
    'prtmbagb', 'prtmbbgr', 'prtmbcgr', 'prtmbhr', 'prtmbahu', 'prtmbbhu', 'prtmbchu', 'prtmbie', 'prtmbaie', 
    'prtmbbie', 'prtmbail', 'prtmbbil', 'prtmblt', 'prtmblv', 'prtmbnl', 'prtmbbnl', 'prtmbcnl', 'prtmbno', 
    'prtmbano', 'prtmbbpl', 'prtmbcpl', 'prtmbdpl', 'prtmbapt', 'prtmbbpt', 'prtmbro', 'prtmbaro', 'prtmbru', 
    'prtmbaru', 'prtmbbru', 'prtmbse', 'prtmbase', 'prtmbbsi', 'prtmbcsi', 'prtmbask', 'prtmbbsk', 'prtmbatr', 
    'prtmbaua', 'prtmbbua', 'prtmbcua'
]

# Populate country-specific party variable descriptions automatically
for var in country_party_variables:
    country_code = var[-2:].upper()
    if var.startswith("prtv"):
        ess_column_descriptions[var] = f"Party voted for in last national election ({country_code})"
    elif var.startswith("prtcl"):
        ess_column_descriptions[var] = f"Party feel closer to ({country_code})"
    elif var.startswith("prtmb"):
        ess_column_descriptions[var] = f"Party member of ({country_code})"

ess_column_descriptions
# %% cell 6
len(ess_column_descriptions)

# %% cell 7
# append the rest of the variable descriptions to the ess_column_descriptions dictionary
ess_column_descriptions.update({
    # Self-reported Health, Well-being & Social Life
    "happy": "How happy are you (0-10)",
    "health": "Subjective general health status (1-5)",
    "hlthhmp": "Hampered in daily activities by illness/disability/mental problem",
    "sclmeet": "How often socially meet with friends, relatives or colleagues",
    "sclact": "Take part in social activities compared to others of same age",
    "inmdisc": "Anyone to discuss intimate/personal matters with",
    "inprdsc": "Number of people with whom you can discuss intimate matters",

    # Identity, Discrimination & Belonging
    "blgetmg": "Belong to a minority ethnic group in country",
    "facntr": "Father born in country",
    "fbrncnt": "Father born in country (older metric)",
    "fbrncnta": "Father born in country (revised scale A)",
    "fbrncntb": "Father born in country (revised scale B)",
    "fbrncntc": "Father born in country (revised scale C)",
    "mocntr": "Mother born in country",
    "mbrncnt": "Mother born in country (older metric)",
    "mbrncnta": "Mother born in country (revised scale A)",
    "mbrncntb": "Mother born in country (revised scale B)",
    "mbrncntc": "Mother born in country (revised scale C)",
    "brncntr": "Respondent born in country",
    "cntbrtha": "Country of birth (alpha code)",
    "cntbrthb": "Country of birth (extended code B)",
    "cntbrthc": "Country of birth (extended code C)",
    "cntbrthd": "Country of birth (extended code D)",
    "ctzcntr": "Citizen of country",
    "ctzshipa": "Country of citizenship (alpha code A)",
    "ctzshipb": "Country of citizenship (alpha code B)",
    "ctzshipc": "Country of citizenship (alpha code C)",
    "ctzshipd": "Country of citizenship (alpha code D)",
    "livecntr": "Year first came to live in country",
    "livecnta": "Year first came to live in country (revised format)",
    "lnghoma": "First language spoken at home",
    "lnghom1": "First language spoken at home (numerical code)",
    "lnghomb": "Second language spoken at home",
    "lnghom2": "Second language spoken at home (numerical code)",
    "dscrgrp": "Member of a group discriminated against in country",
    "dscrdk": "Discrimination ground: Don't know",
    "dscrdsb": "Discrimination ground: Disability",
    "dscretn": "Discrimination ground: Ethnicity",
    "dscrgnd": "Discrimination ground: Gender",
    "dscrlng": "Discrimination ground: Language",
    "dscrna": "Discrimination ground: Not applicable",
    "dscrnap": "Discrimination ground: Not available",
    "dscrntn": "Discrimination ground: Nationality",
    "dscroth": "Discrimination ground: Other",
    "dscrrce": "Discrimination ground: Race",
    "dscrref": "Discrimination ground: Refusal",
    "dscrrlg": "Discrimination ground: Religion",
    "dscrsex": "Discrimination ground: Sexual orientation",
    "dscrage": "Discrimination ground: Age",
    "atchctr": "How emotionally attached to country",
    "atcherp": "How emotionally attached to Europe",

    # Crime & Personal Safety
    "aesfdrk": "Feeling of safety walking alone in local area after dark",
    "crmvct": "Respondent or household member victim of burglary/assault in last 5 years",
    "crvctef": "Victim of crime: physical assault",
    "crvctwr": "Victim of crime: threat or harassment",

    # Religion & Morality
    "rlgblg": "Belong to particular religion or denomination",
    "rlgblge": "Ever belonged to particular religion or denomination",
    "rlgdgr": "How religious are you (0-10)",
    "rlgatnd": "How often attend religious services apart from special occasions",
    "pray": "How often pray apart from religious services",
    "rlgdnm": "Religion or denomination belonging to at present",
    "rlgdnme": "Religion or denomination belonged to in past",

    # Democracy, EU & Institutional Views
    "ccnthum": "Climate change caused by human activity",
    "ccrdprs": "Personal responsibility to reduce climate change",
    "wrclmch": "How worried about climate change",
    "vteurmmb": "Vote in referendum: Country should remain member of EU",
    "vteubcmb": "Vote in referendum: Country should become member of EU",
    "vteumbgb": "Vote in referendum: UK remaining member of EU",

    # Respondent Demographics & Education
    "age": "Respondent's age in years",
    "agea": "Respondent's age calculated from birth year",
    "agegroup": "Age group category",
    "yrbrn": "Respondent's year of birth",
    "gndr": "Gender of respondent",
    "eduyrs": "Years of full-time education completed",
    "edctn": "Highest level of education",
    "edctnp": "Partner's highest level of education",
    "edufld": "Field of education",
    "edulvla": "Highest level of education (ES-ISCED 5-category)",
    "edulvlb": "Highest level of education (detailed ISCED)",
    "eisced": "Highest level of education (European ISCED scheme)",
    "eiscedf": "Father's highest level of education (European ISCED)",
    "eiscedm": "Mother's highest level of education (European ISCED)",
    "eiscedp": "Partner's highest level of education (European ISCED)",
    "domicil": "Type of residential area (big city to rural)",
    "anctry1": "First ancestry code",
    "anctry2": "Second ancestry code",
    "anctrya1": "First ancestry code (revised format)",
    "anctrya2": "Second ancestry code (revised format)",

    # Employment & Household Economy
    "pdwrk": "Paid work in last 7 days",
    "edctn": "In education in last 7 days",
    "uempla": "Unemployed and actively looking for job in last 7 days",
    "uempli": "Unemployed, wanting job but not actively looking in last 7 days",
    "dsbld": "Permanently sick or disabled in last 7 days",
    "rtrd": "Retired in last 7 days",
    "cmsrv": "In community or military service in last 7 days",
    "hswrk": "Doing housework or looking after home in last 7 days",
    "dngoth": "Other main activity in last 7 days",
    "mainact": "Main activity in last 7 days",
    "mnactic": "Main activity in last 7 days (consolidated code)",
    "crpdwk": "Control over paid work schedule",
    "wkhct": "Total contracted hours per week in main job",
    "wkhtot": "Total hours normally worked per week in main job including overtime",
    "emplrel": "Employment relation (employee, self-employed, working for family)",
    "emplno": "Number of employees respondent has",
    "jbspv": "Supervisory responsibility in current job",
    "njbspv": "Number of employees supervised",
    "isco08": "Occupation ISCO-08 code",
    "nacer2": "Industry sector (NACE Rev. 2 classification)",
    "tporgwk": "Type of organization worked for (public/private)",
    "estsz": "Establishment size (number of employees at workplace)",
    "uemp12m": "Ever been unemployed and seeking work for 12 months or more",
    "uemp3m": "Ever been unemployed and seeking work for 3 months or more",
    "uemp5yr": "Any period of unemployment in last 5 years",

    # Income & Household Dynamics
    "hinctnt": "Household net income (deciles/categories)",
    "hinctnta": "Household net income (standardized deciles)",
    "hincfel": "Feeling about household's income nowadays (coping on income)",
    "hincsrca": "Main source of household income",
    "hhmmb": "Number of people living in household",
    "maritala": "Legal marital status",
    "maritalb": "Legal marital status (revised format)",
    "marsts": "Legal marital status including cohabitation",
    "rshpsts": "Relationship status with partner",
    "partner": "Living with husband/wife/partner in household",
    "chldhm": "Children living in household",
    "chldhhe": "Ever had children living in household",

    # Human Values (Schwartz Value Scale)
    "ipcrtiv": "Important to think new ideas and being creative (0-6)",
    "imprich": "Important to be rich, have money and expensive things (0-6)",
    "ipeqopt": "Important that people are treated equally / equal opportunities (0-6)",
    "ipshabt": "Important to show abilities and be admired (0-6)",
    "impsafe": "Important to live in secure surroundings (0-6)",
    "ipadvnt": "Important to seek adventures and take risks (0-6)",
    "ipbhprp": "Important to behave properly and avoid doing wrong things (0-6)",
    "ipudrst": "Important to understand different people and opinions (0-6)",
    "ipmodst": "Important to be humble and modest (0-6)",
    "ipgdtim": "Important to have a good time and spoil oneself (0-6)",
    "iprspot": "Important to make own decisions and be free (0-6)",
    "iphlppl": "Important to help people around and care for others (0-6)",
    "ipsuces": "Important to be successful and recognized (0-6)",
    "ipstrgv": "Important that government ensures safety against threats (0-6)",
    "ipseekv": "Important to look for fun and things that give pleasure (0-6)",
    "ipfrule": "Important to follow rules and obey laws (0-6)",
    "iplylfr": "Important to be loyal to friends and devote to near ones (0-6)",
    "impenv": "Important to care for nature and environment (0-6)",
    "imptrad": "Important to follow traditions and customs (0-6)",
    "impfun": "Important to seek fun and things that bring pleasure (0-6)",
    "impdiff": "Important to try different new things in life (0-6)",
    "impfree": "Important to make own decisions and be independent (0-6)",

    # Region Identifiers
    "region": "Region of residence (NUTS classification)",
    "regunit": "Regional unit aggregation level"
}
)

# --- Dynamic Population for Repeating ESS Variable Families ---

# 1. Household Roster Variables (up to 24 household members)
for i in range(2, 25):
    ess_column_descriptions[f"gndr{i}"] = f"Gender of household member {i}"
    ess_column_descriptions[f"yrbrn{i}"] = f"Year of birth of household member {i}"
    ess_column_descriptions[f"rshipa{i}"] = f"Relationship of household member {i} to respondent"

# 2. Partner Employment & Education Variables (suffix 'p')
partner_vars = {
    "crpdwkp": "Partner: control over paid work schedule",
    "wkhtotp": "Partner: total normal weekly hours worked",
    "emplrelp": "Partner: employment status (employee/self-employed)",
    "emplnop": "Partner: number of employees",
    "jbspvp": "Partner: supervisory responsibility",
    "njbspvp": "Partner: number of employees supervised",
    "isco08p": "Partner: occupation ISCO-08 code",
    "mnactp": "Partner: main activity in last 7 days",
    "rtrdp": "Partner: retired status",
    "dsbldp": "Partner: disabled status",
    "hswrkp": "Partner: housework status"
}
ess_column_descriptions.update(partner_vars)

# 3. Country-Specific Religion Variables (rlgdn* / rlgde*)
# E.g., rlgdnde = Current religion in Germany, rlgdegb = Past religion in Great Britain
relig_vars = [c for c in [
    'rlgdnal', 'rlgdnat', 'rlgdnbat', 'rlgdnbe', 'rlgdnch', 'rlgdnach', 'rlgdncy', 'rlgdnde', 'rlgdnade', 
    'rlgdndk', 'rlgdnfi', 'rlgdnafi', 'rlgdngb', 'rlgdngr', 'rlgdnagr', 'rlgdnhu', 'rlgdnie', 'rlgdnil', 
    'rlgdnis', 'rlgdnais', 'rlgdnlt', 'rlgdnlv', 'rlgdnnl', 'rlgdnno', 'rlgdnpl', 'rlgdnapl', 'rlgdnpt', 
    'rlgdnro', 'rlgdnru', 'rlgdnaru', 'rlgdnse', 'rlgdnase', 'rlgdnsi', 'rlgdnsk', 'rlgdnask', 'rlgdnua', 
    'rlgdnrs', 'rlgdme', 'rlgdeal', 'rlgdeat', 'rlgdebat', 'rlgdebe', 'rlgdech', 'rlgdeach', 'rlgdecy', 
    'rlgdede', 'rlgdeade', 'rlgdedk', 'rlgdefi', 'rlgdeafi', 'rlgdegb', 'rlgdegr', 'rlgdeagr', 'rlgdehu', 
    'rlgdeie', 'rlgdeil', 'rlgdeis', 'rlgdeais', 'rlgdelt', 'rlgdelv', 'rlgdenl', 'rlgdeno', 'rlgdepl', 
    'rlgdeapl', 'rlgdept', 'rlgdero', 'rlgderu', 'rlgdearu', 'rlgdese', 'rlgdease', 'rlgdesi', 'rlgdesk', 
    'rlgdeask', 'rlgdeua', 'rlgders', 'rlgdeme'
]]

for var in relig_vars:
    country_code = var[-2:].upper()
    if 'rlgdn' in var:
        ess_column_descriptions[var] = f"Current religious denomination ({country_code})"
    else:
        ess_column_descriptions[var] = f"Past religious denomination ({country_code})"

# 4. Country-Specific Education Variables (edlv*)
# Tracks education levels for respondent (edlv*), partner (edlvp*), father (edlvf*), mother (edlvm*)
edu_prefixes = [
    ('edlvp', 'Partner country-specific educational attainment'),
    ('edlvf', 'Father country-specific educational attainment'),
    ('edlvm', 'Mother country-specific educational attainment'),
    ('edlv',  'Respondent country-specific educational attainment')
]

# Quick pattern matching for education codes
for var_name in list(set([
    'edagegb', 'edlvat', 'edlveat', 'edlvbe', 'edlvabe', 'edlvdbe', 'edlvebe', 'edlvbg', 'edlvdbg', 
    'edlvebg', 'edlvbch', 'edlvcch', 'edlvdch', 'edlvcy', 'edlvacy', 'edlvdcy', 'edlvecy', 'edlvgcy', 
    'edlvcz', 'edlvdcz', 'edlvdal', 'edlvade', 'edlvdfi', 'edlvdis', 'edlvadk', 'edlvddk', 'edlvdxk', 
    'edlvaee', 'edlvbee', 'edlvdee', 'edlvaes', 'edlvdes', 'edlvees', 'edlvges', 'edlvafr', 'edlvbfr', 
    'edlvdfr', 'edlvgb', 'edlvagr', 'edlvdgr', 'edlvhr', 'edlvdhr', 'edlvehr', 'edlvahu', 'edlvbhu', 
    'edlvdhu', 'edlvaie', 'edlvbie', 'edlvdie', 'edlvail', 'eduil1', 'eduail1', 'edubil1', 'edlvdit', 
    'edlveit', 'edlvlt', 'edlvdlt', 'edlvlv', 'edlvdlv', 'edlvnl', 'edlvdnl', 'edlvenl', 'edlvno', 
    'edlvano', 'edlvdno', 'edlvapl', 'edlvbpl', 'edlvdpl', 'edlvepl', 'edlvgpl', 'edlvapt', 'edlvbpt', 
    'edlvdpt', 'edlvro', 'edlvru', 'edlvdru', 'edlvase', 'edlvdse', 'edlvsi', 'edlvasi', 'edlvdsi', 
    'edlvesi', 'edlvsk', 'edlvask', 'edlvdsk', 'edlvtr', 'edlvua', 'edlvaua', 'edlvdua', 'edude1', 
    'eduade1', 'edubde1', 'edude2', 'eduade2', 'edude3', 'eduade3', 'edugb2', 'eduagb2', 'edubgb2', 
    'eduil2', 'eduail2', 'edupl2', 'eduyrpl', 'edugb1', 'eduagb1', 'edubgb1', 'educgb1', 'edlvdrs', 
    'edlvdme', 'edagepgb', 'edlvpat', 'edlvpeat', 'edlvpbe', 'edlvpdbe', 'edlvpebe', 'edlvpch', 
    'edlvpdch', 'edlvpcy', 'edlvpdcy', 'edlvpecy', 'edlvpgcy', 'edlvpcz', 'edlvpdcz', 'edlvpdbg', 
    'edlvpebg', 'edlvpdfi', 'edlvpdis', 'edlvpdit', 'edlvpeit', 'edlvpdk', 'edlvpddk', 'edlvpdxk', 
    'edlvpee', 'edlvpdee', 'edlvpes', 'edlvpdes', 'edlvpees', 'edlvpfes', 'edlvpfr', 'edlvpdfr', 
    'edlvpgb', 'edlvpgr', 'edlvpdgr', 'edlvphr', 'edlvpdhr', 'edlvpehr', 'edlvphu', 'edlvpdhu', 
    'edlvpie', 'edlvpdie', 'edlvpil', 'edupil1', 'edupail1', 'edupbil1', 'edlvplt', 'edlvpdlt', 
    'edlvplv', 'edlvpdlv', 'edlvpnl', 'edlvpdnl', 'edlvpenl', 'edlvpno', 'edlvpdno', 'edlvppl', 
    'edlvpdpl', 'edlvpepl', 'edlvpfpl', 'edlvppt', 'edlvpdpt', 'edlvpro', 'edlvpru', 'edlvpdru', 
    'edlvpse', 'edlvpdse', 'edlvpsi', 'edlvpdsi', 'edlvpesi', 'edlvpsk', 'edlvpdsk', 'edlvptr', 
    'edlvpua', 'edlvpdua', 'edupde1', 'edupade1', 'edupbde1', 'edupde2', 'edupade2', 'edupde3', 
    'edupade3', 'edupgb2', 'edupagb2', 'edupbgb2', 'edupil2', 'edupail2', 'eduppl2', 'eduyrppl', 
    'edlvpdal', 'edupgb1', 'edupagb1', 'edupbgb1', 'edupcgb1', 'edlvpdrs', 'edlvpdme', 'edlvfdal', 
    'edlvfat', 'edlvfeat', 'edlvfbe', 'edlvfdbe', 'edlvfebe', 'edlvfdbg', 'edlvfebg', 'edlvfhr', 
    'edlvfdhr', 'edlvfehr', 'edlvfcy', 'edlvfdcy', 'edlvfecy', 'edlvfgcy', 'edlvfcz', 'edlvfdcz', 
    'edlvfdk', 'edlvfddk', 'edlvfee', 'edlvfdee', 'edlvfdfi', 'edlvffr', 'edlvfdfr', 'edufde1', 
    'edufade1', 'edufbde1', 'edufde2', 'edufade2', 'edufde3', 'edufade3', 'edlvfgr', 'edlvfdgr', 
    'edlvfhu', 'edlvfdhu', 'edlvfdis', 'edlvfie', 'edlvfdie', 'edlvfil', 'edufil1', 'edufail1', 
    'edufbil1', 'edufil2', 'edufail2', 'edlvfdit', 'edlvfeit', 'edlvfdxk', 'edlvflv', 'edlvfdlv', 
    'edlvflt', 'edlvfdlt', 'edlvfnl', 'edlvfdnl', 'edlvfenl', 'edlvfno', 'edlvfdno', 'edlvfpl', 
    'edlvfdpl', 'edlvfepl', 'edlvffpl', 'edlvfpt', 'edlvfdpt', 'edlvfro', 'edlvfru', 'edlvfdru', 
    'edlvfsk', 'edlvfdsk', 'edlvfsi', 'edlvfdsi', 'edlvfesi', 'edlvfes', 'edlvfdes', 'edlvfees', 
    'edlvffes', 'edlvfse', 'edlvfdse', 'edlvfch', 'edlvfdch', 'edlvftr', 'edlvfua', 'edlvfdua', 
    'edlvfgb', 'edufgb1', 'edufagb1', 'edufbgb1', 'edufcgb1', 'edufgb2', 'edufagb2', 'edufbgb2', 
    'edagefgb', 'edlvfdrs', 'edlvfdme', 'edagemgb', 'edlvmat', 'edlvmeat', 'edlvmbe', 'edlvmdbe', 
    'edlvmebe', 'edlvmch', 'edlvmdch', 'edlvmcy', 'edlvmdcy', 'edlvmecy', 'edlvmgcy', 'edlvmcz', 
    'edlvmdcz', 'edlvmdal', 'edlvmdfi', 'edlvmdis', 'edlvmdit', 'edlvmeit', 'edlvmdk', 'edlvmddk', 
    'edlvmdxk', 'edlvmee', 'edlvmdee', 'edlvmes', 'edlvmdes', 'edlvmees', 'edlvmfes', 'edlvmfr', 
    'edlvmdfr', 'edlvmgb', 'edlvmgr', 'edlvmdgr', 'edlvmhr', 'edlvmdhr', 'edlvmehr', 'edlvmhu', 
    'edlvmdhu', 'edlvmie', 'edlvmdie', 'edlvmil', 'edumil1', 'edumail1', 'edumbil1', 'edlvmlt', 
    'edlvmdlt', 'edlvmlv', 'edlvmdlv', 'edlvmnl', 'edlvmdnl', 'edlvmenl', 'edlvmno', 'edlvmdno', 
    'edlvmpl', 'edlvmdpl', 'edlvmepl', 'edlvmfpl', 'edlvmpt', 'edlvmdpt', 'edlvmro', 'edlvmru', 
    'edlvmdru', 'edlvmse', 'edlvmdse', 'edlvmsi', 'edlvmdsi', 'edlvmesi', 'edlvmsk', 'edlvmdsk', 
    'edlvmtr', 'edlvmua', 'edlvmdua', 'edumde1', 'edumade1', 'edumbde1', 'edumde2', 'edumade2', 
    'edumde3', 'edumade3', 'edumgb1', 'edumagb1', 'edumbgb1', 'edumcgb1', 'edumgb2', 'edumagb2', 
    'edumbgb2', 'edumil2', 'edumail2', 'edlvmdbg', 'edlvmebg', 'edlvmdrs', 'edlvmdme'
])):
    country_code = var_name[-2:].upper()
    if 'edlvp' in var_name or 'edup' in var_name:
        ess_column_descriptions[var_name] = f"Partner country-specific educational level ({country_code})"
    elif 'edlvf' in var_name or 'eduf' in var_name:
        ess_column_descriptions[var_name] = f"Father country-specific educational level ({country_code})"
    elif 'edlvm' in var_name or 'edum' in var_name:
        ess_column_descriptions[var_name] = f"Mother country-specific educational level ({country_code})"
    else:
        ess_column_descriptions[var_name] = f"Respondent country-specific educational level ({country_code})"
len(ess_column_descriptions)

# %% cell 8
ess_column_descriptions

# %% cell 9
# Save the dictionary as csv
ess_column_descriptions_df = pd.DataFrame(list(ess_column_descriptions.items()), columns=['Variable', 'Description'])
ess_column_descriptions_df.to_csv('ess_column_descriptions.csv', index=False)
# %%
# 1. Filter rows to only include rounds 3 through 9
df_sub = df[df['essround'].between(3, 9)]

# 2. Identify essential grouping/metadata columns you always want to keep
# Add any other core identifiers here (e.g., 'idno', 'psu', 'stratum', 'weight')
id_cols = ['cntry', 'essround'] 

# 3. Find columns within rounds 3-9 that have NO missing values
# .isna().any() returns True if a column has at least one NaN
complete_cols = [
    col for col in df_sub.columns 
    if not df_sub[col].isna().any()
]

# 4. Filter the original DataFrame to keep only those columns
filtered_df = df[complete_cols].copy()

# Summary statistics
print(f"Original column count: {df.shape[1]}")
print(f"Filtered column count: {filtered_df.shape[1]}")

# %%
# list all the columns in the filtered DataFrame
filtered_columns = filtered_df.columns.tolist()
filtered_columns