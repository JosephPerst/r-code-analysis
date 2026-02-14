
library(ggpubr)
library(ggplot2)
library(tidyr)
library(dplyr)

data=read.csv("Sentiment and Demographics.csv")

SummaryStats_mm_cond_1=data %>% group_by(mm_cond_1) %>% summarise(sum_sent_educ=sum(Values))

data2=read.csv("Sentiment Demographics WL.csv")
SummaryStats_wh_cur_maint_yn=data2%>% group_by(wh_cur_maint_yn) %>% summarise(sum_rec=sum(Values))

SummaryStats_wh_q22=data2%>% group_by(wh_q22) %>% summarise(sum_rec=sum(Values))
