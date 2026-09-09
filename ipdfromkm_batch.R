"""IPDfromKM batch reconstruction v2: single-read design + coxph for two-arm panels."""
# IPDfromKM 批量重建 v2：一次性读入全部输入（避免沙箱文件视图翻转），内存中逐臂处理
# 用法: Rscript ipdfromkm_batch.R <r_input_dir> <out_dir>
suppressMessages(library(IPDfromKM))
suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]; out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "r_ipd"), showWarnings = FALSE, recursive = TRUE)

longdf <- read.csv(file.path(in_dir, "allarms_long.csv"), stringsAsFactors = FALSE)
riskdf <- read.csv(file.path(in_dir, "risks_long.csv"), stringsAsFactors = FALSE)
keys <- unique(longdf$key)
cat("arms to process:", length(keys), "\n")

summary_rows <- list()
all_ipd <- list()
n_ok <- 0; n_fail <- 0
for (key in keys) {
  datf <- longdf[longdf$key == key, c("time", "surv")]
  rk <- riskdf[riskdf$key == key, c("trisk", "nrisk")]
  if (nrow(datf) < 5 || nrow(rk) < 1) { n_fail <- n_fail + 1; next }
  res <- try({
    pre <- preprocess(dat = datf, trisk = rk$trisk, nrisk = rk$nrisk, maxy = 1)
    getIPD(pre, armID = 1, tot.events = NULL)
  }, silent = TRUE)
  if (inherits(res, "try-error")) {
    n_fail <- n_fail + 1
    summary_rows[[length(summary_rows) + 1]] <- data.frame(key = key, status = "ERROR",
      n_ipd = NA, n_events = NA, rmse = NA, mean_ae = NA, max_ae = NA)
    next
  }
  ipd <- res$IPD
  ipd$key <- key
  all_ipd[[length(all_ipd) + 1]] <- ipd
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    key = key, status = "OK", n_ipd = nrow(ipd),
    n_events = sum(ipd$status == 1),
    rmse = res$precision[["RMSE"]], mean_ae = res$precision[["mean_abserror"]],
    max_ae = res$precision[["max_abserror"]])
  n_ok <- n_ok + 1
}
sumdf <- do.call(rbind, summary_rows)
write.csv(sumdf, file.path(out_dir, "r_summary.csv"), row.names = FALSE)
alldf <- do.call(rbind, all_ipd)
write.csv(alldf, file.path(out_dir, "r_ipd_all.csv"), row.names = FALSE)

# 两臂面板 Cox（直接用内存中的重建 IPD）
suppressMessages(library(survival))
okkeys <- sumdf$key[sumdf$status == "OK"]
panels <- sub("__arm[0-9]+$", "", okkeys)
cox_rows <- list()
for (pp in unique(panels)) {
  ks <- okkeys[panels == pp]
  if (length(ks) != 2) next
  ipd_list <- lapply(ks, function(k) all_ipd[[which(sapply(all_ipd, function(x) x$key[1] == k))]])
  if (any(sapply(ipd_list, nrow) == 0)) next
  for (i in seq_along(ipd_list)) names(ipd_list[[i]])[2] <- "status"
  ipd <- rbind(cbind(ipd_list[[1]][, c("time", "status")], arm = 1),
               cbind(ipd_list[[2]][, c("time", "status")], arm = 2))
  res <- try({
    fit <- coxph(Surv(time, status) ~ arm, data = ipd)
    s <- summary(fit)
    c(HR = s$conf.int[1, "exp(coef)"], lo = s$conf.int[1, "lower .95"],
      hi = s$conf.int[1, "upper .95"], p = as.numeric(s$logtest[3]))
  }, silent = TRUE)
  if (inherits(res, "try-error")) next
  cox_rows[[length(cox_rows) + 1]] <- data.frame(panel = pp, HR = res[["HR"]],
    lo = res[["lo"]], hi = res[["hi"]], p = as.numeric(res[["p"]]))
}
coxdf <- do.call(rbind, cox_rows)
if (!is.null(coxdf)) write.csv(coxdf, file.path(out_dir, "r_cox.csv"), row.names = FALSE)
cat(sprintf("R batch done: ok=%d fail=%d cox_pairs=%d\n", n_ok, n_fail, nrow(coxdf)))
