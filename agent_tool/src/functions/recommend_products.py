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
    log.info(f"recommend_products function started with input: {input}")

    try:
        # Get all sales data
        sales_data = await lookupSales(LookupSalesInput(category="any"))

        # Ensure sales_data is correctly formatted
        if not hasattr(sales_data, "sales") or not sales_data.sales:
            log.error("lookupSales did not return expected sales data")
            return LookupSalesOutput(sales=[])

        # Log all sales data before filtering
        log.info(f"lookupSales returned {len(sales_data.sales)} items: {sales_data.sales}")

        # Fix: Ensure correct filtering logic
        filtered_items = [
            item for item in sales_data.sales
            if item.type in input.preferred_categories and item.sale_price_usd <= input.max_budget
        ]

        # Log filtered results
        log.info(f"Filtered {len(filtered_items)} items: {filtered_items}")

        return LookupSalesOutput(sales=filtered_items)

    except Exception as e:
        log.error(f"Error in recommend_products: {e}")
        return LookupSalesOutput(sales=[])
