| Family | Estimator | ATT estimate (USD) | 95% bootstrap interval | covers benchmark |
|---|---|---|---|---|
| naive | Difference in means | -15,205 | [-16,325, -13,656] | no |
| outcome | OLS regression adjustment | 1,513 | [-123, 3,284] | yes |
| outcome | GBM regression adjustment | 1,496 | [-389, 3,008] | yes |
| propensity | IPW (logistic) | 2,889 | [668, 4,859] | yes |
| propensity | PSM (1-NN, caliper) | 2,764 | [230, 4,159] | yes |
| propensity | PSM (5-NN, caliper) | 2,339 | [323, 4,297] | yes |
| doubly robust | AIPW (ridge + logistic) | 4,390 | [1,843, 9,328] | no |
| doubly robust | AIPW (GBM + GBM) | 2,419 | [240, 3,937] | yes |
| meta-learner | T-learner (GBM) | 1,493 | [-392, 3,007] | yes |
| meta-learner | X-learner (GBM) | 2,262 | [233, 3,926] | yes |
| meta-learner | R-learner (GBM) | 2,047 | [238, 4,289] | yes |
| meta-learner | DR-learner (GBM) | 7,290 | [-2,409, 11,422] | yes |
| doubly robust | DML, partially linear (GBM) | -959 | [-2,622, 775] | no |
