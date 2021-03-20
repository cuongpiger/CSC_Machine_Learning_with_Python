        k-value from 2 to 9:
            build model KNN with k-value (run 10 times)
            score_at_k-value = mean of 10 results
            push score_at_k-value to lst_mean

        get k-value which mean(abs(r^2(train) - r^2(test))) is maximum