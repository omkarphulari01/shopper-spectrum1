# Data

Place the dataset here as `online_retail.csv`.

## Expected columns

| Column | Description |
|---|---|
| InvoiceNo | Transaction number (invoices starting with `C` are cancellations) |
| StockCode | Unique product/item code |
| Description | Product name |
| Quantity | Number of products purchased |
| InvoiceDate | Date and time of the transaction |
| UnitPrice | Price per product |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

## Note

The raw CSV (~48 MB) is excluded from git by default via `.gitignore` to keep
the repo light. If you want to commit it directly, remove the
`data/online_retail.csv` line from `.gitignore`. Alternatively, host it with
[Git LFS](https://git-lfs.com/) or link to the original download in this file.
