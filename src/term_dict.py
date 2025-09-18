# similarity threshold
similarity_threshold = {
    "REVENUE": 0.5,
    "LOSS": 0.4,
    "INDUSTRY":0.2
}


# Revenue-related keywords
REVENUE_KEYWORDS = [
    "total revenue",
    "total revenues", 
    "net revenue",
    "net revenues",
    "total net sales",
    "revenues",
    "revenue",
    "net sales",
    "total sales"
]

# Loss-related keywords
LOSS_KEYWORDS = [
    "total losses",
    "total loss",
    "net losses",
    "net loss", 
    "total net losses",
    "net income (loss)",
    "loss",
    "losses"
]

# Industry-related keywords
INDUSTRY_KEYWORDS = [
    "industry",
    "sector",
    "line business",
    "business category",
    "SIC",
    "NAICS",
    "business activity"
]


# Main dictionary for term matching
TERM_DICT = {
    "REVENUE": REVENUE_KEYWORDS,
    "LOSS": LOSS_KEYWORDS,
    "INDUSTRY": INDUSTRY_KEYWORDS
}
