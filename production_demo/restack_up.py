import os
from restack_sdk_cloud import RestackCloud
from dotenv import load_dotenv  
load_dotenv()

async def main():
    # Initialize the RestackCloud client with the CLOUD token from environment variables
    restack_cloud_client = RestackCloud(os.getenv('RESTACK_CLOUD_TOKEN'))

    engine = {
        'name': 'restack_engine',
        'image': 'ghcr.io/restackio/restack:main',
        'portMapping': [
            {
                'port': 5233,
                'path': '/',
                'name': 'engine-frontend',
            },
            {
                'port': 6233,
                'path': '/api',
                'name': 'engine-api',
            }
        ],
        'environmentVariables': [
          {
              'name': 'RESTACK_ENGINE_ID',
              'value': os.getenv('RESTACK_ENGINE_ID'),
          },
          {
              'name': 'RESTACK_ENGINE_ADDRESS',
              'value': os.getenv('RESTACK_ENGINE_ADDRESS'),
          },
          {
              'name': 'RESTACK_ENGINE_API_KEY',
              'value': os.getenv('RESTACK_ENGINE_API_KEY'),
          },
        ],
    }

    # Define the application configuration
    backend = {
        'name': 'backend',
        'dockerFilePath': '/production_demo/Dockerfile',
        'dockerBuildContext': './production_demo/',
        'environmentVariables': [
            {
                'name': 'RESTACK_ENGINE_ID',
                'value': os.getenv('RESTACK_ENGINE_ID'),
            },
            {
                'name': 'RESTACK_ENGINE_ADDRESS',
                'value': os.getenv('RESTACK_ENGINE_ADDRESS'),
            },
            {
                'name': 'RESTACK_ENGINE_API_KEY',
                'value': os.getenv('RESTACK_ENGINE_API_KEY'),
            },
        ],
    }

    # Configure the stack with the applications
    await restack_cloud_client.stack({
        'name': 'production_demo',
        'previewEnabled': False,
        'applications': [engine,backend],
    })

    # Deploy the stack
    await restack_cloud_client.up()

# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())