# Data

```
data/
├── raw/         # place online_retail.csv here (input)
└── processed/   # generated parquet files (output of `shopper-spectrum train`)
```

## Expected raw columns

| Column | Description |
|---|---|
| InvoiceNo | Transaction number (`C`-prefixed = cancellation) |
| StockCode | Unique product code |
| Description | Product name |
| Quantity | Units purchased |
| InvoiceDate | Transaction datetime |
| UnitPrice | Price per unit |
| CustomerID | Customer identifier |
| Country | Customer country |

The raw CSV (~48 MB) is gitignored by default to keep the repo light. Remove the
`data/raw/*.csv` line from `.gitignore` to commit it, or host it via Git LFS.
