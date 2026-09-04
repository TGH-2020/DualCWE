packages <- c("ape")
install.packages(setdiff(packages, rownames(installed.packages()))) 
library(ape)

get_arg <- function(args, flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) > 0 && idx[1] < length(args)) return(args[idx[1] + 1])
  return(default)
}

args      <- commandArgs(trailingOnly = TRUE)
rooted_tree <- read.tree(get_arg(args, "--rooted_tree", "out/inferred_tree_cosine_dist_upgma.txt"))
out_dir <- get_arg(args, "--out_dir", "out/")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --- Normalize Tree Depth to 1 (avoids scaling issues) ---
max_depth <- max(node.depth.edgelength(rooted_tree))
if (max_depth <= 0) stop("Invalid tree: non-positive depth")
rooted_tree$edge.length <- rooted_tree$edge.length / max_depth

start_time <- Sys.time()
ultra_tree <- chronopl(rooted_tree, lambda=10)
end_time <- Sys.time()

cat("Chronopl completed in", round(difftime(end_time, start_time, units = "mins"), 1), "minutes\n")

# --- Optional: Save ultrametric tree ---
write.tree(ultra_tree, file = file.path(out_dir, "worldtree.ultrametric.nwk"))
