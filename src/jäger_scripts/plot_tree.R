## World Tree Visualization
## Branches colored by family (top families); outer ring = macroarea
## Output: data/worldtree_plot.pdf, data/worldtree_plot.svg

packages <- c("ape", "ggtree", "ggtreeExtra", "ggplot2", "dplyr", "readr")
install.packages(setdiff(packages, rownames(installed.packages()))) 
library(ape)
library(ggtree)
library(ggtreeExtra)
library(ggplot2)
library(dplyr)
library(readr)

get_arg <- function(args, flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) > 0 && idx[1] < length(args)) return(args[idx[1] + 1])
  return(default)
}

args      <- commandArgs(trailingOnly = TRUE)
wordlist_file <- get_arg(args, "--wordlist", "data/lexibank_pruned_wordlist.csv")
tree_file <- get_arg(args, "--world_tree", "out/inferred_tree_cosine_upgma.txt")
lang_file <- get_arg(args, "--lang_file", "data/all_languages.csv")
out_dir <- get_arg(args, "--out_dir", "results/")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ── 1. Load tree ─────────────────────────────────────────────────────────────
tree <- read.tree(tree_file)
cat("Tree loaded:", length(tree$tip.label), "tips\n")

# ── 2. Macroarea mapping ──────────────────────────────────────────────────────
macro_df <- read.csv(lang_file, stringsAsFactors = FALSE)
macro_df <- macro_df[!duplicated(macro_df$glottocode), ]
# Rename columns to match ggtree tip labels
macro_df <- rename(macro_df, Glottocode = glottocode, Macroarea = macroarea)

if (all(c("Glottocode", "Macroarea") %in% names(macro_df))) {
  macro_df <- macro_df[macro_df$Glottocode != "" & macro_df$Macroarea != "", c("Glottocode", "Macroarea")]
} else {
  macro_df <- NULL
}

# ── 3. Family mapping ─────────────────────────────────────────────────────────
wordlist <- read.csv(wordlist_file, stringsAsFactors = FALSE)
fam_df   <- unique(wordlist[, c("Glottocode", "Family")])
fam_df   <- fam_df[fam_df$Glottocode != "" & fam_df$Family != "", ]
fam_df   <- fam_df[!duplicated(fam_df$Glottocode), ]

# ── 4. Tip metadata ───────────────────────────────────────────────────────────
tips <- data.frame(label = tree$tip.label, stringsAsFactors = FALSE)
tips <- left_join(tips, macro_df, by = c("label" = "Glottocode"))
tips <- left_join(tips, fam_df,   by = c("label" = "Glottocode"))
tips$Macroarea[is.na(tips$Macroarea)] <- "Unknown"
tips$Family[is.na(tips$Family)]       <- "Other"
cat("Tips with macroarea:", sum(tips$Macroarea != "Unknown"), "/", nrow(tips), "\n")

# ── 5. Family colors (top 25 families, rest = grey) ──────────────────────────
fam_counts <- sort(table(tips$Family[tips$Family != "Other"]), decreasing = TRUE)
top_fams   <- names(head(fam_counts, 12))
cat("Top families:\n"); print(head(fam_counts, 12))

# 12 clearly distinct colors (ColorBrewer "Paired")
paired_cols <- c("#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C",
                 "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00",
                 "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928")
fam_palette   <- setNames(paired_cols, top_fams)
fam_color_vec <- c(fam_palette, "Other" = "#888888", "0" = "#888888")

# Recode tips: families outside top 25 → "Other"
tips$FamilyGroup <- ifelse(tips$Family %in% top_fams, tips$Family, "Other")

# ── 6. Color branches by family via bottom-up traversal ──────────────────────
# A node is colored with family X only if ALL descendants are in family X.
# This prevents phantom coloring at high-level mixed nodes.
n_tips  <- length(tree$tip.label)
n_total <- n_tips + tree$Nnode
node_fam <- rep("0", n_total)

# Assign tips
for (i in seq_len(n_tips)) {
  gc  <- tree$tip.label[i]
  row <- tips[tips$label == gc, "FamilyGroup"]
  if (length(row) > 0 && !is.na(row[1])) node_fam[i] <- row[1]
}

# Bottom-up: internal nodes (process in reverse postorder)
edge   <- tree$edge
for (nd in rev(seq(n_tips + 1, n_total))) {
  children       <- edge[edge[, 1] == nd, 2]
  child_fams     <- unique(node_fam[children])
  # Ignore unassigned children for propagation
  named_fams     <- child_fams[child_fams != "0" & child_fams != "Other"]
  if (length(named_fams) == 1 && length(child_fams) == 1) {
    node_fam[nd] <- named_fams[1]   # all children same family
  } else {
    node_fam[nd] <- "0"
  }
}

# Attach group attribute to tree (same format as groupOTU output)
tree_g <- tree
attr(tree_g, "group") <- setNames(
  factor(node_fam, levels = c(top_fams, "Other", "0")),
  c(tree$tip.label, as.character(seq(n_tips + 1, n_total)))
)

# ── 7. Macroarea color palette ────────────────────────────────────────────────
area_colors <- c(
  "Africa"        = "#1B9E77",   # teal
  "Eurasia"       = "#D95F02",   # orange
  "Papunesia"     = "#7570B3",   # purple
  "North America" = "#E7298A",   # pink
  "South America" = "#66A61E",   # green
  "Australia"     = "#E6AB02",   # brown-yellow
  "Unknown"       = "#AAAAAA"
)

# ── 8. Build plot ─────────────────────────────────────────────────────────────
p <- ggtree(tree_g, aes(color = group),
            layout = "fan", open.angle = 15,
            size = 0.25) %<+% tips +
  scale_color_manual(
    values = fam_color_vec,
    name   = "Language family",
    breaks = c(top_fams, "Other"),   # exclude "0" (mixed internal nodes)
    guide  = guide_legend(
      title.theme  = element_text(size = 20, face = "bold"),
      label.theme  = element_text(size = 18),
      keywidth     = unit(0.8, "cm"),
      keyheight    = unit(0.4, "cm"),
      ncol         = 1,
      override.aes = list(linewidth = 3)
    )
  )

# Outer ring: macroarea
p <- p +
  geom_fruit(
    geom    = geom_tile,
    mapping = aes(y = label, fill = Macroarea),
    width   = 0.09,
    offset  = 0.08,
    color   = NA
  ) +
  scale_fill_manual(
    values = area_colors,
    name   = "Macroarea",
    guide  = guide_legend(
      title.theme  = element_text(size = 20, face = "bold"),
      label.theme  = element_text(size = 18),
      keywidth     = unit(0.5, "cm"),
      keyheight    = unit(0.5, "cm"),
      override.aes = list(color = NA)
    )
  )

# ── 9. Title and theme ────────────────────────────────────────────────────────
p <- p +
  labs(title = NULL, subtitle = NULL) +
  theme(
    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA),
    plot.margin       = margin(0, 0, 0, 0),
    legend.box           = "vertical",
    legend.background    = element_rect(fill = alpha("white", 0.85), color = NA),
    legend.margin        = margin(6, 8, 6, 8)
  )

# Minimal expansion around the circular tree
p <- p +
  hexpand(0.01, direction =  1) + hexpand(0.01, direction = -1) +
  vexpand(0.01, direction =  1) + vexpand(0.01, direction = -1)

# ── 10. Save ──────────────────────────────────────────────────────────────────
ggsave(file.path(out_dir, "worldtree_plot.pdf"), p, width = 18, height = 18, units = "in")
ggsave(file.path(out_dir, "worldtree_plot.png"), p, width = 18, height = 18, units = "in", dpi = 200)
cat("Saved", file.path(out_dir, "worldtree_plot.pdf"), "\n")
