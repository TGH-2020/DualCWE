# src/get_full_trees.R
#
# Infer BIONJ and UPGMA trees from a language distance matrix and prune the
# Glottolog tree to the same language set.
#
# Usage:
#   Rscript src/get_full_trees.R \
#     --dist_csv  out/cosine_dist.csv \
#     --glottolog data/glottolog.tre \
#     --out_dir   results/
#
# Output files (all in --out_dir):
#   inferred_tree_<dist_name>_nj.txt
#   inferred_tree_<dist_name>_upgma.txt
#   pruned_glottolog_tree.txt

packages <- c("ape", "phangorn")
install.packages(setdiff(packages, rownames(installed.packages()))) 
library(ape)
library(phangorn)

# Argument parsing
get_arg <- function(args, flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) > 0 && idx[1] < length(args)) return(args[idx[1] + 1])
  return(default)
}

args      <- commandArgs(trailingOnly = TRUE)
dist_csv  <- get_arg(args, "--dist_csv", "out/cosine_dist.csv")
glottolog <- get_arg(args, "--glottolog", "data/glottolog.tre")
out_dir   <- get_arg(args, "--out_dir", "results/")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Load data
distMatrix    <- as.matrix(read.csv(dist_csv, row.names = 1))
glottologTree <- read.tree(glottolog)

dist_name <- tools::file_path_sans_ext(basename(dist_csv))

# BIONJ (Neighbor Joining)
nj_tree  <- bionj(distMatrix)
nj_path  <- file.path(out_dir, paste0("inferred_tree_", dist_name, "_nj.txt"))
write.tree(nj_tree, file = nj_path)
cat(sprintf("Saved NJ tree    -> %s\n", nj_path))

# UPGMA
upgma_tree  <- upgma(distMatrix)
upgma_path  <- file.path(out_dir, paste0("inferred_tree_", dist_name, "_upgma.txt"))
write.tree(upgma_tree, file = upgma_path)
cat(sprintf("Saved UPGMA tree -> %s\n", upgma_path))

# Prune Glottolog tree to the language set in the distance matrix
tips_to_drop  <- setdiff(glottologTree$tip.label, nj_tree$tip.label)
prunedTree    <- drop.tip(glottologTree, tips_to_drop)
glottolog_out <- file.path(out_dir, "pruned_glottolog_tree.txt")
write.tree(prunedTree, file = glottolog_out)
cat(sprintf("Saved pruned Glottolog tree -> %s\n", glottolog_out))
cat(sprintf("Language set: %d languages\n", length(nj_tree$tip.label)))
