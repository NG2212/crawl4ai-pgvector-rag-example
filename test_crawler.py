import asyncio
from crawl4ai import AsyncWebCrawler

async def test_crawler():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"Testing crawler with URL: {url}")
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            
            # Print all available attributes
            print("\nAvailable attributes:")
            for attr in dir(result):
                if not attr.startswith('_'):  # Skip private attributes
                    value = getattr(result, attr)
                    if not callable(value):  # Skip methods
                        print(f"{attr}: {type(value)}")
                        if isinstance(value, str):
                            print(f"Length: {len(value)}")
                            print(f"Preview: {value[:100]}...")

    except Exception as e:
        print(f"Error during crawling: {e}")

if __name__ == "__main__":
    asyncio.run(test_crawler()) 