from interpret.glassbox import ExplainableBoostingClassifier

#ebm = ExplainableBoostingClassifier()
#ebm.fit(X_train, y_train)

# or substitute with LogisticRegression, DecisionTreeClassifier, RuleListClassifier, ...
# EBM supports pandas dataframes, numpy arrays, and handles "string" data natively.