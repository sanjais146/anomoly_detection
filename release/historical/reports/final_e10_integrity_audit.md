# E10 Final Test Integrity Audit
- **Feature Generation:** Test transactions correctly generated historical windows using exactly the same $t_{history} < t_{target}$ filtering and 7-day $\Delta$ delay constraints as training.
- **Data Isolation:** `X_tr` explicitly used `tr` and `va` indices. Test labels (`y_te`) were ONLY used in final metric calculation `f1_score(y_te, preds)`. No early stopping or tuning occurred against Test.
- **Threshold Policy:** Evaluated precisely at the frozen validation-optimum of `0.4040`. Threshold tuning post-Test-visibility was explicitly barred.
