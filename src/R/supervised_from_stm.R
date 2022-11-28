library(textrecipes)
library(tidymodels)
library(parsnip)
library(rsample)
library(yardstick)
library(dplyr)
library(LiblineaR)
library(kernlab)

gafamfit2 <- stm(out$documents, out$vocab, K=20, prevalence=~company_search+s(year), 
                max.em.its=75, data=out$meta, init.type="Spectral", 
                seed=28)

labelTopics(gafamfit2)

png(file.path(dir, "/plots/stm_top_topics10.png"), width = 1000, height = 800)
plot.STM(gafamfit,type="summary", n = 10 ,text.cex=1.2, width=80, height=90, xlim=c(0,.8)) #cex.main =1
dev.off()

png(file.path(dir, "/plots/stm_top_topics10_unique.png"), width = 1000, height = 800)
plot.STM(gafamfit,type="summary", label="frex", n = 10 ,text.cex=1.2, width=80, height=90, xlim=c(0,.8)) #cex.main =1
dev.off()

plot(gafamfit, type="labels", topics=c(4,2,8,3,6))
plot(gafamfit, type="labels", topics=c(7,1,10,9,5))

doc_topic <- tidy(gafamfit2, matrix = "theta") %>%
  left_join(out$meta, by="document")

write.csv(doc_topic, file.path(dir, "data/stm_doc_topic10.csv"))

data <- doc_topic %>%
  #filter(!topic %in% c(9,5,4)) %>%
  filter(topic %in% c(4,8,3,7)) %>%
  group_by(document) %>%
  mutate(max_gamma = max(gamma)) %>%
  ungroup() %>%
  filter(max_gamma == gamma) %>%
  mutate(label = as.factor(topic))
#  filter(topic %in% c()) %>%
#  filter(gamma>=0.5)


# topic 4** - EU competition investigation in general
# topic 2 - Microsoft cloud acquisition
# topic 8** - US FTC investigation on big tech industries brought to senate
# topic 3** - data privacy regulation, DMA, Facebook
# topic 6 - patent antitrust court cases
# topic 7** - competition enforcement on platform and innovative markets
# topic 1** - advertising google search engine/ browser
# topic 10** - app/game store payment system (Apple vs Fortnite Epic games)
# topic 9 - online retail market in India (Amazon, Flipkart)
# topic 5 - 

  
df_split <- initial_split(data, prop=0.8)
train_data <- training(df_split)
test_data <- testing(df_split)

rec <-recipe(label ~ text, data = train_data)

rec <- rec %>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text)

model <- svm_linear(mode="classification")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(model)

model_fit <- wf %>% 
  fit(train_data)

test_data$prediction <- predict(model_fit, test_data)$.pred_class

scorer(test_data, truth=label, estimate=prediction, event_level="second")

# tp <- sum((test_data$env==1)&(test_data$prediction==1))
# fn <- sum((test_data$env==1)&(test_data$prediction==0))
# fp <- sum((test_data$env==0)&(test_data$prediction==1))
# tn <- sum((test_data$env==0)&(test_data$prediction==0))
# recall <- tp / (tp+fn)
# precision <- tp / (tp+fp)
# 
# scorer <- metric_set(
#   yardstick::accuracy, 
#   yardstick::precision, 
#   yardstick::recall,
#   yardstick::f_meas
# )

##------------------------ HYPERPARAMETER TUNING ------------------------##

model <- svm_poly(cost=tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(model)

folds <- vfold_cv(train_data, v = 2)
svm_res <- tune_grid(
  wf, resamples = folds, grid = 2,
  metrics = metric_set(f_meas),
  control = control_grid(event_level="second")
)
collect_metrics(svm_res)

best_model <- svm_res %>% select_best()

final_workflow <- wf %>% 
  finalize_workflow(best_model)


