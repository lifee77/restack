# src/functions/recommend_products.py
from typing import List
from restack_ai.function import function, log
from pydantic import BaseModel
from src.functions.lookup_sales import lookupSales, LookupSalesInput, LookupSalesOutput

class RecommendationInput(BaseModel):
    user_id: int
    preferred_categories: List[str]
    max_budget: float

@function.defn()
async def recommend_products(input: RecommendationInput) -> LookupSalesOutput:
    log.info("recommend_products function started", input=input)
    # Get all sales data
    sales_data = await lookupSales(LookupSalesInput(category="any"))
    # Filter items based on preferred categories and budget
    filtered_items = [
        item 
        for item in sales_data.sales 
        if item.type in input.preferred_categories and item.sale_price_usd <= input.max_budget
    ]
    return LookupSalesOutput(sales=filtered_items)
