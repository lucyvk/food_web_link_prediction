# This file runs regressions between properties of the food webs and missing link prediction performance, overall and split up by ecosystem type.
# Author: Lucy Van Kleunen
#
# Run with R version 4.3.1

library(tidyverse) # version: 2.0.0
library(reshape2) # version: 1.4.4
library(ggpmisc) # version: 0.5.4-1
library(dotwhisker) # version: 0.8.2
library(dplyr) # version: 1.1.2
library(glue) # version: 1.6.2
library(ggpubr) # version: 0.6.0
library(corrplot) # version: 0.92

model_colors <- c('#6B8E23','#5f9ea0','#D2691E')
model_shapes <- c(6,2,1)
FONT_SIZE <- 9
FONT_SIZE2 <- 15.75

# To set the fonts for the figures
library(extrafont)
windowsFonts(sans="Helvetica")
loadfonts(device="win")
loadfonts(device="postscript")
loadfonts(device = "pdf")

to_star_string <- function(p) {
  if (p <= 0.0001) {
    return("****")
  } else if (p <= 0.001){
    return("***")
  } else if (p <= 0.01){
    return("**")
  } else if (p <= 0.05) {
    return("*")
  } else {
    return("")
  }
}

res_location = 'Summarized_Results/food_web_lp_res.csv'

# number display settings 
options(scipen=100, digits=5)

network_metadata <- read.csv('Processed_Data_Disaggregated_Lifestage/food_web_metadata.csv', header = TRUE, sep = ",", dec = ".")
display_names <- read.csv('net_props_display_names.csv', header = TRUE, sep = ",", dec = ".")
display_dict <- setNames(display_names$display, display_names$feat)

# get a list of network ids in each ecosystem type
all_ids <- network_metadata$fw_id
terrestrial_aboveground_ids <- filter(network_metadata, ecosystem.type=="terrestrial aboveground")$fw_id
marine_ids <- filter(network_metadata, ecosystem.type=="marine")$fw_id
terrestrial_belowground_ids <- filter(network_metadata, ecosystem.type=="terrestrial belowground")$fw_id
lakes_ids <- filter(network_metadata, ecosystem.type=="lakes")$fw_id
streams_ids <- filter(network_metadata, ecosystem.type=="streams")$fw_id

eco_type_ids <- list(all_ids,terrestrial_aboveground_ids,marine_ids,terrestrial_belowground_ids,lakes_ids,streams_ids)
eco_type_names <- c("all","terrestrial_aboveground","marine","terrestrial_belowground","lakes","streams")
eco_display <- setNames(c("All","Terrestrial Above","Marine","Terrestrial Below","Lakes","Streams"),eco_type_names)

for (typ in 1:6) {
  
  curr_eco_type_ids <- eco_type_ids[[typ]]
  curr_eco_type_name <- eco_type_names[typ]
  print(curr_eco_type_name)

  food_web_lp_res <- read.csv(res_location, header = TRUE, sep = ",", dec = ".")
  food_web_lp_res <-  food_web_lp_res %>% arrange(net_id)
  
  # Filter to just the network ids relevant to this ecosystem type
  food_web_lp_res <- filter(food_web_lp_res, net_id %in% curr_eco_type_ids)
  
  # All the numeric covariates to do initial univariate linear regressions with
  covariates <- colnames(network_metadata)
  covariates <- covariates[-which(covariates %in% c('X','fw_id','ecosystem.type','reciprocated_edge'))]
  
  network_metadata_curr <- filter(network_metadata, fw_id %in% curr_eco_type_ids)
  
  # if column constant, remove it and don't do that regression for this ecosystem type
  names_to_remove <- list()
  for (col in covariates){
    if (sd(network_metadata_curr[[col]], na.rm=TRUE) == 0){
      print("constant col:")
      print(col)
      names_to_remove <- append(names_to_remove, list(col))
    }
  }
  for (col in names_to_remove){
    covariates <- covariates[covariates != col]
  }
  
  # Scale so we can compare coefficient estimates between features on different scales
  # Note - to reproduce the unscaled version of the plots, adjust to skip this step
  covariates_scaled <- as.data.frame(scale(network_metadata_curr[,covariates]))
  df <- cbind(food_web_lp_res, network_metadata_curr[,covariates])
  df_scaled <- cbind(food_web_lp_res, covariates_scaled)
  
  # EXAMPLE (unscaled)
  ex <- lm(struc_roc_mean ~ log_num_nodes, data=df)
  summary(ex)
  dwplot(ex)
  
  # EXAMPLE (scaled)
  ex <- lm(struc_roc_mean ~ log_num_nodes, data=df_scaled)
  summary(ex)
  dwplot(ex)
  
  # EXAMPLE (num nodes + other)
  ex <- lm(struc_roc_mean ~ log_num_nodes + avg_degree, data=df_scaled)
  summary(ex)
  
  struc_roc_formulas <- sapply(covariates, function(x) as.formula(paste('struc_roc_mean ~', x)))
  attr_roc_formulas <- sapply(covariates, function(x) as.formula(paste('attr_roc_mean ~', x)))
  full_roc_formulas <- sapply(covariates, function(x) as.formula(paste('full_roc_mean ~', x)))
  
  struc_pr_formulas <- sapply(covariates, function(x) as.formula(paste('struc_pr_mean ~', x)))
  attr_pr_formulas <- sapply(covariates, function(x) as.formula(paste('attr_pr_mean ~', x)))
  full_pr_formulas <- sapply(covariates, function(x) as.formula(paste('full_pr_mean ~', x)))
  
  # Run a version of the correlations with log_num_nodes included
  if (curr_eco_type_name == "all"){
    
    covariates_trimmed <- covariates[covariates != 'log_num_nodes'] # look at all other covariates
  
    struc_roc_formulas_logN <- sapply(covariates_trimmed, function(x) as.formula(paste('struc_roc_mean ~ log_num_nodes + ', x)))
    attr_roc_formulas_logN <- sapply(covariates_trimmed, function(x) as.formula(paste('attr_roc_mean ~ log_num_nodes + ', x)))
    full_roc_formulas_logN <- sapply(covariates_trimmed, function(x) as.formula(paste('full_roc_mean ~ log_num_nodes + ', x)))
    
    struc_pr_formulas_logN <- sapply(covariates_trimmed, function(x) as.formula(paste('struc_pr_mean ~ log_num_nodes + ', x)))
    attr_pr_formulas_logN <- sapply(covariates_trimmed, function(x) as.formula(paste('attr_pr_mean ~ log_num_nodes + ', x)))
    full_pr_formulas_logN <- sapply(covariates_trimmed, function(x) as.formula(paste('full_pr_mean ~ log_num_nodes + ', x)))
  
  }
  
  struc_roc_models <- lapply(struc_roc_formulas, function(x){lm(x, data = df_scaled)})
  attr_roc_models <- lapply(attr_roc_formulas, function(x){lm(x, data = df_scaled)})
  full_roc_models <- lapply(full_roc_formulas, function(x){lm(x, data = df_scaled)})
  
  struc_pr_models <- lapply(struc_pr_formulas, function(x){lm(x, data = df_scaled)})
  attr_pr_models <- lapply(attr_pr_formulas, function(x){lm(x, data = df_scaled)})
  full_pr_models <- lapply(full_pr_formulas, function(x){lm(x, data = df_scaled)})
  
  if (curr_eco_type_name == "all"){
    
    struc_roc_models_logN <- lapply(struc_roc_formulas_logN, function(x){lm(x, data = df_scaled)})
    attr_roc_models_logN <- lapply(attr_roc_formulas_logN, function(x){lm(x, data = df_scaled)})
    full_roc_models_logN <- lapply(full_roc_formulas_logN, function(x){lm(x, data = df_scaled)})
    
    struc_pr_models_logN <- lapply(struc_pr_formulas_logN, function(x){lm(x, data = df_scaled)})
    attr_pr_models_logN <- lapply(attr_pr_formulas_logN, function(x){lm(x, data = df_scaled)})
    full_pr_models_logN <- lapply(full_pr_formulas_logN, function(x){lm(x, data = df_scaled)})
    
  }
  
  # Unpack all the regression results to write them to file
  
  unpack_result_univariate <- function(models) {
    results <- lapply(models,
           function(x){ 
             x <- summary(x)
             intercept <- x$coefficients[1]
             coef<- x$coefficients[2]
             intercept_se <- x$coefficients[3]
             coef_se <- x$coefficients[4]
             intercept_pvalue <- x$coefficients[7]
             coef_pvalue <- x$coefficients[8]
             mult_r_squared <- x$r.squared
             adjusted_r_squared <- x$adj.r.squared
             res<- as.numeric(c(intercept,intercept_se,coef,coef_se,intercept_pvalue,coef_pvalue,mult_r_squared,adjusted_r_squared))
             names(res)<-c("intercept","intercept_se","coef","coef_se","intercept_pvalue","coef_pvalue","mult_r_squared","adjusted_r_squared")
             return(res)
           })
    res_df <- as.data.frame(t(as.data.frame(results, check.names = FALSE)))
    p_adjusted <- p.adjust(res_df$coef_pvalue, method="BH")
    res_df <- cbind(res_df,p_adjusted)
    res_df <- res_df[order(res_df$p_adjusted),]
    return(res_df)
  }
  
  struc_roc_res_df <- unpack_result_univariate(struc_roc_models)
  attr_roc_res_df <- unpack_result_univariate(attr_roc_models)
  full_roc_res_df <- unpack_result_univariate(full_roc_models)
  struc_pr_res_df <- unpack_result_univariate(struc_pr_models)
  attr_pr_res_df <- unpack_result_univariate(attr_pr_models)
  full_pr_res_df <- unpack_result_univariate(full_pr_models)
  
  if (curr_eco_type_name == "all"){
    
    # assuming log_num_nodes is the first one in the formula (as set up now)
    unpack_result_logN <- function(models) {
      results <- lapply(models,
                        function(x){ 
                          x <- summary(x)
                          intercept <- x$coefficients[1]
                          coef<- x$coefficients[3] # skip log_num_nodes
                          intercept_se <- x$coefficients[4]
                          coef_se <- x$coefficients[6]  # skip log_num_nodes
                          intercept_pvalue <- x$coefficients[10]
                          coef_pvalue <- x$coefficients[12]  # skip log_num_nodes
                          mult_r_squared <- x$r.squared
                          adjusted_r_squared <- x$adj.r.squared
                          res<- as.numeric(c(intercept,intercept_se,coef,coef_se,intercept_pvalue,coef_pvalue,mult_r_squared,adjusted_r_squared))
                          names(res)<-c("intercept","intercept_se","coef","coef_se","intercept_pvalue","coef_pvalue","mult_r_squared","adjusted_r_squared")
                          return(res)
                        })
      res_df <- as.data.frame(t(as.data.frame(results, check.names = FALSE)))
      p_adjusted <- p.adjust(res_df$coef_pvalue, method="BH")
      res_df <- cbind(res_df,p_adjusted)
      res_df <- res_df[order(res_df$p_adjusted),]
      return(res_df)
    }

    struc_roc_res_df_logN <- unpack_result_logN(struc_roc_models_logN)
    attr_roc_res_df_logN <- unpack_result_logN(attr_roc_models_logN)
    full_roc_res_df_logN <- unpack_result_logN(full_roc_models_logN)
    struc_pr_res_df_logN <- unpack_result_logN(struc_pr_models_logN)
    attr_pr_res_df_logN <- unpack_result_logN(attr_pr_models_logN)
    full_pr_res_df_logN <- unpack_result_logN(full_pr_models_logN)
  }
  
  # Make overall coefficient plots for the two different metrics
  custom_coef_plot <- function(coef_df_to_plot, vars_order, titl_type, legend_position, y_lab, x_low, x_high) {
    # display names for features
    vars_order <- unlist(lapply(vars_order, function(x) display_dict[x]))
    coef_df_to_plot$term <- unlist(lapply(coef_df_to_plot$term, function(x) display_dict[x]))
    text_vect <- lapply(filter(coef_df_to_plot, model=="Full")[match(rev(vars_order), coef_df_to_plot$term),]$p.adjusted,to_star_string)
    min_est <- min(coef_df_to_plot$estimate) - 2*max(abs(coef_df_to_plot$std.error))
    max_est <- max(coef_df_to_plot$estimate) + 2*max(abs(coef_df_to_plot$std.error))
    range_est <-max_est-min_est
    curr_plot <- dwplot(coef_df_to_plot, vline = geom_vline(
      xintercept = 0,
      colour = "grey60",
      linetype = 2
    ), vars_order=vars_order, dot_args = list(aes(shape = model))) + ggtitle(glue("Predicting {titl_type}")) +
      labs(y=y_lab,caption="Coefficient estimate (univariate)")+
      theme_bw(base_family="Helvetica") + 
      scale_color_manual(values = model_colors)+
      scale_shape_manual(values = model_shapes)+
      guides(shape = guide_legend("Model"),colour = guide_legend("Model"))+
      xlim(c(NA,max_est+range_est*0.25))+
      theme(
        plot.title.position = "plot", # align with left of plot
        plot.caption.position = "plot", # align with left of plot
        plot.title = element_text(family="Helvetica", size=FONT_SIZE,color="black", hjust=0.5),
        axis.text.x = element_text(family = "Helvetica",size=FONT_SIZE, angle=90,color="black", vjust = 0.5, hjust = 1),
        axis.text.y = element_text(family = "Helvetica",size=FONT_SIZE,color="black"),
        axis.title.y = element_text(family="Helvetica",size =FONT_SIZE,color="black"),
        axis.title.x = element_text(family="Helvetica",size =FONT_SIZE,color="black"),
        legend.title = element_text(family="Helvetica",size =FONT_SIZE,color="black"),
        legend.text= element_text(family="Helvetica",size =FONT_SIZE,color="black")
      )
    add_annotation <- function(xx,yy,txt){
      annotate("text",x=xx,y=yy-0.1,label=txt,size=FONT_SIZE/.pt)
    }
    return(curr_plot + add_annotation(rep(max_est+range_est*0.175,length(text_vect)),seq.int(1,length(text_vect)),text_vect))
  }
  
  # coefficient order - match the order in supplemental tables
  coef_order <- c("miss_mass_frac", "high_tax_frac", "unclassified_tax_frac", "parasitic_removal", "cannibalistic_removal", "num_attr", "min_log_mass","max_log_mass","mean_log_mass","median_log_mass","mass_skewness","log_num_nodes","avg_degree","connectance","clustering_coefficient","modularity","assort_mass","assort_met","assort_mov","deg_assort_io","deg_assort_oi","deg_assort_oo","deg_assort_ii")
  for (col in names_to_remove){
    coef_order <- coef_order[coef_order != col]
  }
  
  # ROC-AUC
  
  # Create a dataframe in the form expected by the dotwhisker plot (look like a multivariate res)
  struc_roc_part <- select(struc_roc_res_df, c("coef","coef_se","p_adjusted"))
  struc_roc_part <- cbind(rownames(struc_roc_part), data.frame(struc_roc_part, row.names=NULL))
  colnames(struc_roc_part) <- c("term", "estimate", "std.error","p.adjusted")
  struc_roc_part$model <- "Structure"
  
  attr_roc_part <- select(attr_roc_res_df, c("coef","coef_se","p_adjusted"))
  attr_roc_part <- cbind(rownames(attr_roc_part), data.frame(attr_roc_part, row.names=NULL))
  colnames(attr_roc_part) <- c("term", "estimate", "std.error","p.adjusted")
  attr_roc_part$model <- "Attribute"
  
  full_roc_part <- select(full_roc_res_df, c("coef","coef_se","p_adjusted"))
  full_roc_part <- cbind(rownames(full_roc_part), data.frame(full_roc_part, row.names=NULL))
  colnames(full_roc_part) <- c("term", "estimate", "std.error","p.adjusted")
  full_roc_part$model <- "Full"
  
  roc_coef_results_plot <- rbind(full_roc_part,attr_roc_part,struc_roc_part)
  roc_coef_panel <- custom_coef_plot(roc_coef_results_plot, coef_order, "ROC-AUC", "none", "Food web properties", -0.15, 0.15)
  
  # PR-AUC
  
  # Create a dataframe in the form expected by the dotwhisker plot (look like a multivariate res)
  struc_pr_part <- select(struc_pr_res_df, c("coef","coef_se","p_adjusted"))
  struc_pr_part <- cbind(rownames(struc_pr_part), data.frame(struc_pr_part, row.names=NULL))
  colnames(struc_pr_part) <- c("term", "estimate", "std.error","p.adjusted")
  struc_pr_part$model <- "Structure"
  
  attr_pr_part <- select(attr_pr_res_df, c("coef","coef_se","p_adjusted"))
  attr_pr_part <- cbind(rownames(attr_pr_part), data.frame(attr_pr_part, row.names=NULL))
  colnames(attr_pr_part) <- c("term", "estimate", "std.error","p.adjusted")
  attr_pr_part$model <- "Attribute"
  
  full_pr_part <- select(full_pr_res_df, c("coef","coef_se","p_adjusted"))
  full_pr_part <- cbind(rownames(full_pr_part), data.frame(full_pr_part, row.names=NULL))
  colnames(full_pr_part) <- c("term", "estimate", "std.error","p.adjusted")
  full_pr_part$model <- "Full"
  
  pr_coef_results_plot <- rbind(full_pr_part,attr_pr_part,struc_pr_part)
  pr_coef_panel <- custom_coef_plot(pr_coef_results_plot, coef_order, "PR-AUC", "none", "", -0.15, 0.15)
  
  if (curr_eco_type_name == "all"){
    plot_labels <- c("a","b")
    coef_figure <- ggarrange(roc_coef_panel, pr_coef_panel, labels=plot_labels, ncol=2, nrow=1, legend="bottom", common.legend=TRUE, font.label=list(family="Helvetica", size = FONT_SIZE, face="bold"))
  } else {
    plot_labels <- c("","","a","b")
    margin_adjust <- 7
    if (curr_eco_type_name == "terrestrial_aboveground" | curr_eco_type_name == "terrestrial_belowground"){
      margin_adjust <- 3
    }
    overall_title <- as_ggplot(text_grob(eco_display[curr_eco_type_name],size = FONT_SIZE, family="Helvetica")) + theme(plot.margin = margin(0, margin_adjust,0,0, "cm"))
    coef_figure <- ggarrange(overall_title, NULL, roc_coef_panel, pr_coef_panel, labels=plot_labels, ncol=2, nrow=2, legend="bottom", common.legend=TRUE, heights = c(1,9), font.label=list(family="Helvetica", size = FONT_SIZE, face="bold"))
  }
  ggsave(glue("Regression_Results/Coefficient_Plot_Scaled_{curr_eco_type_name}.pdf"), width=7.4, height=6, device=cairo_pdf)

  # Output a table with the adjusted p-values (a bit more information on results than the coefficient plot)
  roc_coef_results_plot$metric <- "ROC-AUC"
  pr_coef_results_plot$metric <- "PR-AUC"
  full_reg_results <- rbind(roc_coef_results_plot,pr_coef_results_plot)
  colnames(full_reg_results)[1:4] <- c("feature_name","coef_estimate","coef_std_error","coef_p_adjusted")
  full_reg_results$significance <- sapply(full_reg_results$coef_p_adjusted, to_star_string)
  write.csv(full_reg_results, file = glue("Regression_Results/food_web_regression_results_{curr_eco_type_name}.csv"))
  
  if (curr_eco_type_name == "all"){
    
    coef_order <- c("miss_mass_frac", "high_tax_frac", "unclassified_tax_frac", "parasitic_removal", "cannibalistic_removal", "num_attr", "min_log_mass","max_log_mass","mean_log_mass","median_log_mass","mass_skewness","avg_degree","connectance","clustering_coefficient","modularity","assort_mass","assort_met","assort_mov","deg_assort_io","deg_assort_oi","deg_assort_oo","deg_assort_ii")
    
    # Output a table and coefficient plot in the same format for the results with logN included
    
    # ROC-AUC
    
    # Create a dataframe in the form expected by the dotwhisker plot (look like a multivariate res)
    struc_roc_part_logN <- select(struc_roc_res_df_logN, c("coef","coef_se","p_adjusted"))
    struc_roc_part_logN <- cbind(rownames(struc_roc_part_logN), data.frame(struc_roc_part_logN, row.names=NULL))
    colnames(struc_roc_part_logN) <- c("term", "estimate", "std.error","p.adjusted")
    struc_roc_part_logN$model <- "Structure"
    
    attr_roc_part_logN <- select(attr_roc_res_df_logN, c("coef","coef_se","p_adjusted"))
    attr_roc_part_logN <- cbind(rownames(attr_roc_part_logN), data.frame(attr_roc_part_logN, row.names=NULL))
    colnames(attr_roc_part_logN) <- c("term", "estimate", "std.error","p.adjusted")
    attr_roc_part_logN$model <- "Attribute"
    
    full_roc_part_logN <- select(full_roc_res_df_logN, c("coef","coef_se","p_adjusted"))
    full_roc_part_logN <- cbind(rownames(full_roc_part_logN), data.frame(full_roc_part_logN, row.names=NULL))
    colnames(full_roc_part_logN) <- c("term", "estimate", "std.error","p.adjusted")
    full_roc_part_logN$model <- "Full"
    
    roc_coef_results_plot_logN <- rbind(full_roc_part_logN,attr_roc_part_logN,struc_roc_part_logN)
    roc_coef_panel_logN <- custom_coef_plot(roc_coef_results_plot_logN, coef_order, "ROC-AUC", "none", "Food web properties, controlling for log(N)", -0.15, 0.15)
    
    # PR-AUC
    
    # Create a dataframe in the form expected by the dotwhisker plot (look like a multivariate res)
    struc_pr_part_logN <- select(struc_pr_res_df_logN, c("coef","coef_se","p_adjusted"))
    struc_pr_part_logN <- cbind(rownames(struc_pr_part_logN), data.frame(struc_pr_part_logN, row.names=NULL))
    colnames(struc_pr_part_logN) <- c("term", "estimate", "std.error","p.adjusted")
    struc_pr_part_logN$model <- "Structure"
    
    attr_pr_part_logN <- select(attr_pr_res_df_logN, c("coef","coef_se","p_adjusted"))
    attr_pr_part_logN <- cbind(rownames(attr_pr_part_logN), data.frame(attr_pr_part_logN, row.names=NULL))
    colnames(attr_pr_part_logN) <- c("term", "estimate", "std.error","p.adjusted")
    attr_pr_part_logN$model <- "Attribute"
    
    full_pr_part_logN <- select(full_pr_res_df_logN, c("coef","coef_se","p_adjusted"))
    full_pr_part_logN <- cbind(rownames(full_pr_part_logN), data.frame(full_pr_part_logN, row.names=NULL))
    colnames(full_pr_part_logN) <- c("term", "estimate", "std.error","p.adjusted")
    full_pr_part_logN$model <- "Full"
    
    pr_coef_results_plot_logN <- rbind(full_pr_part_logN,attr_pr_part_logN,struc_pr_part_logN)
    pr_coef_panel_logN <- custom_coef_plot(pr_coef_results_plot_logN, coef_order, "PR-AUC", "none", "", -0.15, 0.15)
    
    roc_coef_results_plot_logN$metric <- "ROC-AUC"
    pr_coef_results_plot_logN$metric <- "PR-AUC"
    full_reg_results_logN <- rbind(roc_coef_results_plot_logN, pr_coef_results_plot_logN)
    colnames(full_reg_results_logN)[1:4] <- c("feature_name","coef_estimate","coef_std_error","coef_p_adjusted")
    full_reg_results_logN$significance <- sapply(full_reg_results_logN$coef_p_adjusted, to_star_string)
    write.csv(full_reg_results_logN, file = glue("Regression_Results/food_web_regression_results_{curr_eco_type_name}_logN.csv"))
    
    pdf(file=glue("Regression_Results/Coefficient_Plot_Scaled_{curr_eco_type_name}_logN.pdf"), width=7.4, height=6, family="Helvetica")
    plot_labels <- c("a","b")
    coef_figure <- ggarrange(roc_coef_panel_logN, pr_coef_panel_logN, labels=plot_labels, ncol=2, nrow=1, legend="bottom", common.legend=TRUE, font.label=list(family="Helvetica", size = FONT_SIZE, face="plain"))
    print(coef_figure)
    dev.off()

    # Plot individual covariates vs. metrics (colored by model)
    
    # Create melted version of the full data frame for plotting predictive performance separated by model type
    df$ecosystem.type <- network_metadata$ecosystem.type
    id_vars <- colnames(df)
    id_vars <- id_vars[-which(id_vars %in% c('struc_roc_mean','attr_roc_mean','full_roc_mean','struc_pr_mean','attr_pr_mean','full_pr_mean'))]
    df_melted <- melt(df,id.vars=id_vars,variable.name='model',value.name='metric')
    df_melted_roc <- subset(df_melted, model == 'struc_roc_mean' | model == 'attr_roc_mean' | model == 'full_roc_mean')
    colnames(df_melted_roc) <- unlist(lapply(colnames(df_melted_roc), function(x){if (x %in% names(display_dict)){return(display_dict[x])} else {return(x)}}))
    df_melted_pr <- subset(df_melted, model == 'struc_pr_mean' | model == 'attr_pr_mean' | model == 'full_pr_mean')
    colnames(df_melted_pr) <- unlist(lapply(colnames(df_melted_pr), function(x){if (x %in% names(display_dict)){return(display_dict[x])} else {return(x)}}))
    
    struc_roc_ps <- struc_roc_res_df$p_adjusted
    names(struc_roc_ps) <- unlist(lapply(rownames(struc_roc_res_df), function(x){if (x %in% names(display_dict)){return(display_dict[x])} else {return(x)}}))
    
    attr_roc_ps <- attr_roc_res_df$p_adjusted
    names(attr_roc_ps) <- unlist(lapply(rownames(attr_roc_res_df), function(x){if (x %in% names(display_dict)){return(display_dict[x])} else {return(x)}}))

    full_roc_ps <- full_roc_res_df$p_adjusted
    names(full_roc_ps) <- unlist(lapply(rownames(full_roc_res_df), function(x){if (x %in% names(display_dict)){return(display_dict[x])} else {return(x)}}))

    struc_pr_ps <- struc_pr_res_df$p_adjusted
    names(struc_pr_ps) <- unlist(lapply(rownames(struc_pr_res_df), function(x){if (x %in% names(display_dict)){return(display_dict[x])} else {return(x)}}))
    
    attr_pr_ps <- attr_pr_res_df$p_adjusted
    names(attr_pr_ps) <- unlist(lapply(rownames(attr_pr_res_df), function(x){if (x %in% names(display_dict)){return(display_dict[x])} else {return(x)}}))
    
    full_pr_ps <- full_pr_res_df$p_adjusted
    names(full_pr_ps) <- unlist(lapply(rownames(full_pr_res_df), function(x){if (x %in% names(display_dict)){return(display_dict[x])} else {return(x)}}))

    # ROC-AUC plotting
    covariates_disp <- unlist(lapply(covariates, function(x){if (x %in% names(display_dict)){return(display_dict[x])} else {return(x)}}))
    lapply(covariates_disp, 
           function(x) {
             print(x)
             model_labels <- c(paste("Structure", to_star_string(struc_roc_ps[x]), sep=" "), paste("Attribute",to_star_string(attr_roc_ps[x]), sep = " "), paste("Full ",to_star_string(full_roc_ps[x]),sep=" "))
             scatter <- ggplot(df_melted_roc, aes(x=!!sym(x), y=metric, color=model, shape=model)) +
               geom_point() +
               geom_smooth(method = lm, aes(fill=model)) +
               scale_color_manual(values = model_colors, labels=model_labels) +
               scale_fill_manual(values = model_colors, labels=model_labels) +
               scale_shape_manual(values = model_shapes, labels=model_labels) +
               ylim(0,1.2) +
               geom_hline(yintercept=0,linetype=2) +
               geom_hline(yintercept=0.5,linetype=2) +
               geom_hline(yintercept=1,linetype=2) +
               ylab("Average ROC-AUC")+
               ggtitle(sprintf('Linear regression, %s',x))+
               theme_classic(base_family = "Helvetica") +
               theme(
                 plot.title = element_text(family="Helvetica", size=FONT_SIZE2,color="black"),
                 axis.text.x = element_text(family = "Helvetica",size=FONT_SIZE2, angle=90,color="black"),
                 axis.text.y = element_text(family = "Helvetica",size=FONT_SIZE2,color="black"),
                 axis.title.x = element_text(family="Helvetica",size = FONT_SIZE2,color="black"),
                 axis.title.y = element_text(family="Helvetica",size = FONT_SIZE2,color="black"),
                 legend.title = element_text(family="Helvetica",size = FONT_SIZE2,color="black"),
                 legend.text= element_text(family="Helvetica",size = FONT_SIZE2,color="black")
               )
             ggsave(sprintf('Regression_Results/ROC/%s_roc_regression_plot.jpg',x), height=5, width=7)
           })
    
    # ROC-AUC plotting
    lapply(covariates_disp, 
           function(x) {
             print(x)
             scatter <- ggplot(df_melted_roc, aes(x=!!sym(x), y=metric, color=model)) +
               geom_point() +
               geom_smooth(method = lm, aes(fill=model)) +
               scale_color_manual(values = model_colors) +
               scale_fill_manual(values = model_colors) +
               ylim(0,1.2) +
               geom_hline(yintercept=0,linetype=2) +
               geom_hline(yintercept=0.5,linetype=2) +
               geom_hline(yintercept=1,linetype=2) +
               ylab("Average ROC-AUC")+
               ggtitle(sprintf('Linear regression, %s',x))+
               theme_classic(base_family = "Helvetica") +
               facet_wrap(~ ecosystem.type)
             ggsave(sprintf('Regression_Results/ROC/%s_roc_regression_plot_eco_type.jpg',x), height=5, width=7)
           })
    
    # PR-AUC plotting
    lapply(covariates_disp, 
           function(x) {
             print(x)
             model_labels <- c(paste("Structure", to_star_string(struc_pr_ps[x]), sep=" "), paste("Attribute",to_star_string(attr_pr_ps[x]), sep = " "), paste("Full ",to_star_string(full_pr_ps[x]),sep=" "))
             scatter <- ggplot(df_melted_pr, aes(x=!!sym(x), y=metric, color=model, shape=model)) +
               geom_point() +
               geom_smooth(method = lm, aes(fill=model)) +
               scale_color_manual(values = model_colors, labels=model_labels) +
               scale_fill_manual(values = model_colors, labels=model_labels) +
               scale_shape_manual(values = model_shapes, labels=model_labels) +
               ylim(-0.2,1.2) +
               geom_hline(yintercept=0,linetype=2) +
               geom_hline(yintercept=1,linetype=2) +
               ggtitle(sprintf('Linear regression, %s',x))+
               ylab("Average PR-AUC")+
               theme_classic(base_family = "Helvetica")+
               theme(
                 plot.title = element_text(family="Helvetica", size=FONT_SIZE2,color="black"),
                 axis.text.x = element_text(family = "Helvetica",size=FONT_SIZE2, angle=90,color="black"),
                 axis.text.y = element_text(family = "Helvetica",size=FONT_SIZE2,color="black"),
                 axis.title.x = element_text(family="Helvetica",size = FONT_SIZE2,color="black"),
                 axis.title.y = element_text(family="Helvetica",size = FONT_SIZE2,color="black"),
                 legend.title = element_text(family="Helvetica",size = FONT_SIZE2,color="black"),
                 legend.text= element_text(family="Helvetica",size = FONT_SIZE2,color="black")
               )
             ggsave(sprintf('Regression_Results/PR/%s_pr_regression_plot.jpg',x), height=5, width=7)
           })
    
    # PR-AUC plotting
    lapply(covariates_disp, 
           function(x) {
             print(x)
             scatter <- ggplot(df_melted_pr, aes(x=!!sym(x), y=metric, color=model)) +
               geom_point() +
               geom_smooth(method = lm, aes(fill=model)) +
               scale_color_manual(values = model_colors) +
               scale_fill_manual(values = model_colors) +
               ylim(-0.2,1.2) +
               geom_hline(yintercept=0,linetype=2) +
               geom_hline(yintercept=1,linetype=2) +
               ggtitle(sprintf('Linear regression, %s',x))+
               ylab("Average PR-AUC")+
               theme_classic(base_family = "Helvetica") +
               facet_wrap(~ ecosystem.type)
             ggsave(sprintf('Regression_Results/PR/%s_pr_regression_plot_eco_type.jpg',x), height=5, width=7)
           })
    
    # Correlation plot between covariates
    correlations <- cor(df_scaled[,covariates])
    pdf(file="Regression_Results/Feature_Correlation_Plot.pdf", family="Helvetica")
    print(corrplot(correlations, type = "upper", order = "hclust",
             tl.col = "black", tl.srt = 45))
    dev.off()

  }
}

