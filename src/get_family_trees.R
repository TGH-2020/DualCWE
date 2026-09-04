# src/get_family_trees.R
#
# Split a language distance matrix by family, infer BIONJ and UPGMA trees
# for each family, and save the corresponding pruned Glottolog tree.
#
# Usage:
#   Rscript src/get_family_trees.R \
#     --dist_csv       results/cosine_dist.csv \
#     --forms_csv      data/lexibank_pruned_wordlist.csv \
#     --glottolog      data/glottolog.tre \
#     --out_dir        results/families/ \
#     --min_languages  4
#
# Output files (all in --out_dir), one set per qualifying family:
#   <fam_name>_nj.txt
#   <fam_name>_upgma.txt
#   <fam_name>_glottolog.txt   (only when Glottolog has matching tips)
#
# fam_name is the family name sanitized for use in a filename.

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
dist_csv  <- get_arg(args, "--dist_csv", "results/cosine_dist.csv")
forms_csv <- get_arg(args, "--forms_csv", "data/lexibank_pruned_wordlist.csv")
glottolog <- get_arg(args, "--glottolog", "data/glottolog.tre")
out_dir   <- get_arg(args, "--out_dir", "results/families/")
min_langs <- as.integer(get_arg(args, "--min_languages", "4"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Load data

distMatrix    <- as.matrix(read.csv(dist_csv, row.names = 1))
glottologTree <- read.tree(glottolog)

# Build Glottocode -> Family mapping from the forms CSV
forms     <- read.csv(forms_csv, stringsAsFactors = FALSE)
forms     <- forms[!is.na(forms$Glottocode) & !is.na(forms$Family), ]
forms     <- forms[forms$Glottocode %in% rownames(distMatrix), ]
forms     <- forms[!duplicated(forms$Glottocode), ]
lang2fam  <- setNames(forms$Family, forms$Glottocode)

families  <- sort(unique(lang2fam))
cat(sprintf("Distance matrix: %d languages, %d families\n",
            nrow(distMatrix), length(families)))

# Sanitise a string for use as a filename component
safe_name <- function(x) {
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("^_|_$", "", x)
  x
}

# Per-family processing

for (fam in families) {
  fam_langs <- names(lang2fam)[lang2fam == fam]
  fam_langs <- intersect(fam_langs, rownames(distMatrix))

  if (length(fam_langs) < min_langs) {
    cat(sprintf("  Skipping '%s': %d languages < min %d\n",
                fam, length(fam_langs), min_langs))
    next
  }

  fam_safe <- safe_name(fam)
  sub_mat  <- distMatrix[fam_langs, fam_langs]

  # BIONJ
  nj_tree  <- bionj(sub_mat)
  nj_path  <- file.path(out_dir, paste0(fam_safe, "_nj.txt"))
  write.tree(nj_tree, file = nj_path)

  # UPGMA
  upgma_tree <- upgma(sub_mat)
  upgma_path <- file.path(out_dir, paste0(fam_safe, "_upgma.txt"))
  write.tree(upgma_tree, file = upgma_path)

  # Pruned Glottolog tree for this family
  tips_to_drop <- setdiff(glottologTree$tip.label, fam_langs)
  pruned <- tryCatch({
    if (length(tips_to_drop) == length(glottologTree$tip.label)) NULL
    else drop.tip(glottologTree, tips_to_drop)
  }, error = function(e) {
    cat(sprintf("    Glottolog pruning error for '%s': %s\n", fam, e$message))
    NULL
  })

  if (!is.null(pruned) && length(pruned$tip.label) >= 2) {
    glottolog_path <- file.path(out_dir, paste0(fam_safe, "_glottolog.txt"))
    write.tree(pruned, file = glottolog_path)
    cat(sprintf("  %-35s %3d languages  [NJ, UPGMA, Glottolog]\n",
                paste0("'", fam, "'"), length(fam_langs)))
  } else {
    cat(sprintf("  %-35s %3d languages  [NJ, UPGMA] (no Glottolog match)\n",
                paste0("'", fam, "'"), length(fam_langs)))
  }
}

cat("Done.\n")
