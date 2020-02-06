#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
-----------------------------------------------------------------------------------------
                Pfeature_select Program was developed by Phuong Van Nguyen 
                      Email: phuong.nguyen@economics.uni-kiel.de
                            Copyright @ Phuong Van Nguyen
---------------------------------Introduction--------------------------------------------
The procedure for feature selection plays an important role in training a Machine Learning
model. Three roles of this procedure are given as follows. First, it removes
irrelevant feature which might worsen the performance of a Machine Learning model.
Second, it makes the data dimensionality reduction. Third, it provides initial ideas about
the relationship between the target feature and explanatory variables. 

To this end, I create three functions that can apply to any type of data, including
categorical and numerical data. These functions are extremely useful for users. 
This is because one just declare several parameters, he/she can produce the four following
vital results

1. How many features are needed to explain the target feature?
2. Visualize the contribution of each feature to the target feature.
3. Visualize top contributors to the target feature.
4. Visualize the cumulative contribution of all explanatory variables to the target feature.
5. Save the results of contribution of all explanatory variables to the target feature.

A wide range of state-of-the-art Machine Learnings are used, such as 
Principal Component Analysis (PCA), Ensemble methods (both bagging and boosting ones)
-----------------------------------------------------------------------------------------
"""

# In[1]:


def plt_contrib_feature(data,figsize_all,title_contrib):
    """
    data:ETR_selection
    figsize_all: (10,100)
    title_contrib:'The bagged decision tree-based importance of explanatory variables'
    """
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=figsize_all) 
    plt.barh(data['feature_importance'].sort_values(ascending=True).index,
         100*data['feature_importance'].sort_values(ascending=True))
    plt.autoscale(enable=True, axis='both',tight=True)
    plt.title(title_contrib,
        fontsize=13, fontweight='bold')
    plt.ylabel('The explanatory variables',fontsize=15)
    plt.xlabel('The contribution (%)', fontsize=15)
    plt.grid(which='major',linestyle=':',linewidth=0.9)
    for i,v in enumerate(round(100*data['feature_importance'].sort_values(ascending=True),2)):
            ax.text(v , i-0.15 , str(v), color='blue')#, fontweight='bold')  



def plt_top_contrib(data,top_contrib,title_top_contrib):
    """
    data:ETR_select_cum
    top_contrib: integer such as 10, 20, 30 etc
    title_top_contrib:'The Top Ten Important Explanatory Variables'
    """
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(10,5)) 
    plt.barh(data['feature_importance'].head(top_contrib).sort_values(ascending=True).index,
         data['feature_importance'].head(top_contrib).sort_values(ascending=True))
    plt.autoscale(enable=True, axis='both',tight=True)
    plt.title(title_top_contrib,
        fontsize=13, fontweight='bold')
    plt.ylabel('The explanatory variables',fontsize=15)
    plt.xlabel('The contribution (%)', fontsize=15)
    plt.grid(which='major',linestyle=':',linewidth=0.9)
    for i,v in enumerate(round(data['feature_importance'].head(top_contrib).
                           sort_values(ascending=True),2)):
        ax.text(v , i-0.15 , str(v), color='blue')#, fontweight='bold')  
        



def plt_cum(data,threshold_cum,title_cum):
    """
    data: ETR_select_cum
    threshold_cum: integer such as 80, 90, 99
    title_cum:'Cumulative Feature Importance'
    """
    import matplotlib.pyplot as plt
    axisX=range(1,
      len(data.sort_values('cumulative_importance',ascending=True).cumulative_importance)+1)
    plt.figure(figsize=(12,5))
    plt.plot(list(axisX),
    data.sort_values('cumulative_importance',ascending=True).cumulative_importance,
       color='r')
    plt.xlabel('Number of Features', size = 14); 
    plt.ylabel('Cumulative Importance', size = 14)
    plt.title(title_cum, size = 16)
    importance_index =len(data[data['cumulative_importance']<threshold_cum].sort_values('cumulative_importance'
                                                                       ,ascending=True))
    plt.vlines(x = importance_index+1,ymin = data.sort_values('cumulative_importance',ascending=True).cumulative_importance.head(1)
           , ymax = threshold_cum,  linestyles='--', colors = 'blue')
    plt.autoscale(enable=True, axis='x',tight=True) 
    return importance_index;



def Pfeature_selection(X,Y,method,no_estimators,figsize_all,title_contrib,
                      top_contrib,title_top_contrib,
                     threshold_cum,title_cum):
    """
    This program is to choose the feature by using ensemble methods. They are 
    both bagging and boosting algorithms. In particular, in terms of bagging algorithm, they are
    1. Random Forest
    2. Extra Trees.
    On the other hand, boosting algorithm, two algorithms:
    1. Adaptive Boosting.
    2. Gradients Boosting.
    Copyright @ Phuong Van Phuong
    Email: phuong.nguyen@economics.uni-kiel.de
    X: pandas dataframe of all explanatory features
    method: 'ExtraTreesReg'
    n_estimators: a number of estimator
    figsize_all: (10,12)
    title_contrib:'The bagged decision tree-based importance of explanatory variables'
    top_contrib: integer such as 10, 20, 30 etc
    title_top_contrib:'The Top Ten Important Explanatory Variables'
    threshold_cum: integer such as 80, 90, 99
    title_cum:'Cumulative Feature Importance'
    """
    import numpy as np
    import pandas as pd
    # Handling data
    examined_X=X.values
    name_feature=X.columns
    num_component=len(name_feature)
    ################################
    if method=='PCA':
        from sklearn.decomposition import PCA
        # Handling data
        examined_X=X.values
        name_feature=X.columns
        num_component=len(name_feature)
        # Configuring the PCA algorithm
        pca = PCA(n_components=num_component)
        # Fitting the data
        fit_pca = pca.fit(examined_X)
        # Recording the result pca_based_contribution=pca_explained_variance
        PCA_feature_importances= fit_pca.explained_variance_ratio_
        PCA_selection=pd.DataFrame(data={'Feature':name_feature,
                                      'feature_importance':PCA_feature_importances})
        PCA_selection=PCA_selection.set_index(['Feature'])
        # plot contributions of all explanatory variables (plt_contrib_feature)
        plt_contrib_feature(PCA_selection,figsize_all,title_contrib)
        #
        pd.options.display.float_format = '{:.4f}'.format
        PCA_selection_temp=100*PCA_selection.sort_values('feature_importance',ascending=False)
        PCA_selection_temp['cumulative_importance']=np.cumsum(PCA_selection_temp,axis=0)
        PCA_select_cum=PCA_selection_temp
        # plot top N contributors
        plt_top_contrib(PCA_select_cum,top_contrib,title_top_contrib)
        #
        importance_index=plt_cum(PCA_select_cum,threshold_cum,title_cum)
        print('PCA indicates %d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold_cum))
        return PCA_select_cum;  
      
    
    #Bagging Methods
    ## ExtraTrees
    ### ExtraTreesRegressor
    if method=='ExtraTreesReg':
        from sklearn.ensemble import ExtraTreesRegressor
        model_ETR=ExtraTreesRegressor(n_estimators=no_estimators)
        model_ETR.fit(examined_X, Y)
        ETR_feature_importances=model_ETR.feature_importances_
        ETR_selection=pd.DataFrame(data={'Feature':name_feature,
                                      'feature_importance':ETR_feature_importances})
        ETR_selection=ETR_selection.set_index(['Feature'])
        # plot contributions of all explanatory variables (plt_contrib_feature)
        plt_contrib_feature(ETR_selection,figsize_all,title_contrib)
        #
        pd.options.display.float_format = '{:.4f}'.format
        ETR_selection_temp=100*ETR_selection.sort_values('feature_importance',ascending=False)
        ETR_selection_temp['cumulative_importance']=np.cumsum(ETR_selection_temp,axis=0)
        ETR_select_cum=ETR_selection_temp#1[ETR_select1['cumulative_importance']<99]
        # plot top N contributors
        plt_top_contrib(ETR_select_cum,top_contrib,title_top_contrib)
        #
        importance_index=plt_cum(ETR_select_cum,threshold_cum,title_cum)
        print('ExtraTreesRegressor indicates %d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold_cum))
        return ETR_select_cum;  
    
    ### ExtraTreesClassifier
    if method=='ExtraTreesClass':
        from sklearn.ensemble import ExtraTreesClassifier
        model_ETC=ExtraTreesClassifier(n_estimators=no_estimators)
        model_ETC.fit(examined_X, Y)
        ETC_feature_importances=model_ETC.feature_importances_
        ETC_selection=pd.DataFrame(data={'Feature':name_feature,
                                      'feature_importance':ETC_feature_importances})
        ETC_selection=ETC_selection.set_index(['Feature'])
        # plot contributions of all explanatory variables
        plt_contrib_feature(ETC_selection,figsize_all,title_contrib)
        #
        pd.options.display.float_format = '{:.4f}'.format
        ETC_selection_temp=100*ETC_selection.sort_values('feature_importance',ascending=False)
        ETC_selection_temp['cumulative_importance']=np.cumsum(ETR_selection_temp,axis=0)
        ETC_select_cum=ETC_selection_temp#1[ETR_select1['cumulative_importance']<99]
        # plot top N contributors
        plt_top_contrib(ETC_select_cum,top_contrib,title_top_contrib)
        #
        importance_index=plt_cum(ETC_select_cum,threshold_cum,title_cum)
        print('ExtraTreesClassifier indicates %d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold_cum))
        return ETC_select_cum;  
    ## RandomForest
    ### RandomForestRegressor
    if method=='RandomForestReg':
        from sklearn.ensemble import RandomForestRegressor
        model_RFR=RandomForestRegressor(n_estimators=no_estimators)
        model_RFR.fit(examined_X, Y)
        RFR_feature_importances=model_RFR.feature_importances_
        RFR_selection=pd.DataFrame(data={'Feature':name_feature,
                                      'feature_importance':RFR_feature_importances})
        RFR_selection=RFR_selection.set_index(['Feature'])
        # plot contributions of all explanatory variables
        plt_contrib_feature(RFR_selection,figsize_all,title_contrib)
        #
        pd.options.display.float_format = '{:.4f}'.format
        RFR_selection_temp=100*RFR_selection.sort_values('feature_importance',ascending=False)
        RFR_selection_temp['cumulative_importance']=np.cumsum(RFR_selection_temp,axis=0)
        RFR_select_cum=RFR_selection_temp#1[ETR_select1['cumulative_importance']<99]
        # plot top N contributors
        plt_top_contrib(RFR_select_cum,top_contrib,title_top_contrib)
        #
        importance_index=plt_cum(RFR_select_cum,threshold_cum,title_cum)
        print('RandomForestRegressor indicates %d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold_cum))
        return RFR_select_cum;  
    ### RandomForestClassifier
    if method=='RandomForestClass':
        from sklearn.ensemble import RandomForestClassifier
        model_RFC=RandomForestClassifier(n_estimators=no_estimators)
        model_RFC.fit(examined_X, Y)
        RFC_feature_importances=model_RFC.feature_importances_
        RFC_selection=pd.DataFrame(data={'Feature':name_feature,
                                      'feature_importance':RFC_feature_importances})
        RFC_selection=RFC_selection.set_index(['Feature'])
        # plot contributions of all explanatory variables
        plt_contrib_feature(RFC_selection,figsize_all,title_contrib)
        #
        pd.options.display.float_format = '{:.4f}'.format
        RFC_selection_temp=100*RFC_selection.sort_values('feature_importance',ascending=False)
        RFC_selection_temp['cumulative_importance']=np.cumsum(RFC_selection_temp,axis=0)
        RFC_select_cum=RFC_selection_temp#1[ETR_select1['cumulative_importance']<99]
        # plot top N contributors
        plt_top_contrib(RFC_select_cum,top_contrib,title_top_contrib)
        #
        importance_index=plt_cum(RFC_select_cum,threshold_cum,title_cum)
        print('RandomForestRegressor indicates %d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold_cum))
        return RFC_select_cum;  
    
    # Boosting Methods
    ## AdaBoost
    ### AdaBoostRegressor
    if method=='AdaBoostReg':
        from sklearn.ensemble import AdaBoostRegressor
        model_ABR=AdaBoostRegressor(n_estimators=no_estimators)
        model_ABR.fit(examined_X, Y)
        ABR_feature_importances=model_ABR.feature_importances_
        ABR_selection=pd.DataFrame(data={'Feature':name_feature,
                                      'feature_importance':ABR_feature_importances})
        ABR_selection=ABR_selection.set_index(['Feature'])
        # plot contributions of all explanatory variables
        plt_contrib_feature(ABR_selection,figsize_all,title_contrib)
        #
        pd.options.display.float_format = '{:.4f}'.format
        ABR_selection_temp=100*ABR_selection.sort_values('feature_importance',ascending=False)
        ABR_selection_temp['cumulative_importance']=np.cumsum(ABR_selection_temp,axis=0)
        ABR_select_cum=ABR_selection_temp#1[ETR_select1['cumulative_importance']<99]
        # plot top N contributors
        plt_top_contrib(ABR_select_cum,top_contrib,title_top_contrib)
        #
        importance_index=plt_cum(ABR_select_cum,threshold_cum,title_cum)
        print('AdaBoostRegressor indicates %d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold_cum))
        return ABR_select_cum; 
    ### AdaBoostClassifier
    if method=='AdaBoostClass':
        from sklearn.ensemble import AdaBoostClassifier
        model_ABC=AdaBoostClassifier(n_estimators=no_estimators)
        model_ABC.fit(examined_X, Y)
        ABC_feature_importances=model_ABC.feature_importances_
        ABC_selection=pd.DataFrame(data={'Feature':name_feature,
                                      'feature_importance':ABC_feature_importances})
        ABC_selection=ABC_selection.set_index(['Feature'])
        # plot contributions of all explanatory variables
        plt_contrib_feature(ABC_selection,figsize_all,title_contrib)
        #
        pd.options.display.float_format = '{:.4f}'.format
        ABC_selection_temp=100*ABR_selection.sort_values('feature_importance',ascending=False)
        ABC_selection_temp['cumulative_importance']=np.cumsum(ABC_selection_temp,axis=0)
        ABC_select_cum=ABC_selection_temp#1[ETR_select1['cumulative_importance']<99]
        # plot top N contributors
        plt_top_contrib(ABC_select_cum,top_contrib,title_top_contrib)
        #
        importance_index=plt_cum(ABC_select_cum,threshold_cum,title_cum)
        print('AdaBoostClassifier indicates %d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold_cum))
        return ABC_select_cum; 
    ## GradientBoosting
    ### GradientBoostingRegressor
    if method=='GradientBoostingReg':
        from sklearn.ensemble import GradientBoostingRegressor
        model_GBR=GradientBoostingRegressor(n_estimators=no_estimators)
        model_GBR.fit(examined_X, Y)
        GBR_feature_importances=model_GBR.feature_importances_
        GBR_selection=pd.DataFrame(data={'Feature':name_feature,
                                      'feature_importance':GBR_feature_importances})
        GBR_selection=GBR_selection.set_index(['Feature'])
        # plot contributions of all explanatory variables
        plt_contrib_feature(GBR_selection,figsize_all,title_contrib)
        #
        pd.options.display.float_format = '{:.4f}'.format
        GBR_selection_temp=100*GBR_selection.sort_values('feature_importance',ascending=False)
        GBR_selection_temp['cumulative_importance']=np.cumsum(GBR_selection_temp,axis=0)
        GBR_select_cum=GBR_selection_temp#1[ETR_select1['cumulative_importance']<99]
        # plot top N contributors
        plt_top_contrib(GBR_select_cum,top_contrib,title_top_contrib)
        #
        importance_index=plt_cum(GBR_select_cum,threshold_cum,title_cum)
        print('GradientBoostingRegressor indicates %d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold_cum))
        return GBR_select_cum; 
    ### GradientBoostingClassifier
    if method=='GradientBoostingClass':
        from sklearn.ensemble import GradientBoostingClassifier
        model_GBC=GradientBoostingClassifier(n_estimators=no_estimators)
        model_GBC.fit(examined_X, Y)
        GBC_feature_importances=model_GBC.feature_importances_
        GBC_selection=pd.DataFrame(data={'Feature':name_feature,
                                      'feature_importance':GBC_feature_importances})
        GBC_selection=GBC_selection.set_index(['Feature'])
        # plot contributions of all explanatory variables
        plt_contrib_feature(GBC_selection,figsize_all,title_contrib)
        #
        pd.options.display.float_format = '{:.4f}'.format
        GBC_selection_temp=100*GBC_selection.sort_values('feature_importance',ascending=False)
        GBC_selection_temp['cumulative_importance']=np.cumsum(GBC_selection_temp,axis=0)
        GBC_select_cum=GBC_selection_temp#1[ETR_select1['cumulative_importance']<99]
        # plot top N contributors
        plt_top_contrib(GBC_select_cum,top_contrib,title_top_contrib)
        #
        importance_index=plt_cum(GBC_select_cum,threshold_cum,title_cum)
        print('GradientBoostingClassifier indicates %d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold_cum))
        return GBC_select_cum; 
    
    else: 
        print('Hello')

