import json
import logging
import signal
import sys
from concurrent.futures import TimeoutError

from google.cloud import pubsub_v1

from app.core.config import configs
from app.service.face_processing_service import FaceProcessingService

logger = logging.getLogger(__name__)


class PubSubSubscriber:
    """
    Pull messages from the PubSub subscription 'banking-ekyc-sign-up-sub'.

    Expected message format:
        {
            "event": "user_ekyc_completed",
            "email": "user@example.com",
            "timestamp": "2026-02-19T12:00:00+00:00"
        }

    On receiving a message:
      1. Parse the email from the message
      2. Delegate to FaceProcessingService to download images → extract embeddings → store in DB
      3. ACK or NACK the message based on success/failure
    """

    def __init__(self, face_processing_service: FaceProcessingService) -> None:
        self._face_processing_svc = face_processing_service
        self._project_id = configs.GCP_PROJECT_ID
        self._subscription_id = configs.PUBSUB_SUBSCRIPTION
        self._subscriber = pubsub_v1.SubscriberClient()
        self._subscription_path = self._subscriber.subscription_path(
            self._project_id, self._subscription_id
        )
        self._streaming_pull_future = None
        logger.info(
            f"PubSubSubscriber initialized — subscription: {self._subscription_path}"
        )

    def _handle_message(self, message: pubsub_v1.subscriber.message.Message) -> None:
        """Process a single PubSub message."""
        logger.info(f"Received message: {message.message_id}")

        try:
            data = json.loads(message.data.decode("utf-8"))
            event = data.get("event")
            email = data.get("email")

            if event != "user_ekyc_completed":
                logger.warning(f"Ignoring unknown event: {event}")
                message.ack()
                return

            if not email:
                logger.error("Message missing 'email' field, acknowledging anyway")
                message.ack()
                return

            logger.info(f"Processing eKYC completed event for: {email}")
            success = self._face_processing_svc.process_user(email)

            if success:
                logger.info(f"Successfully processed embeddings for: {email}")
                message.ack()
            else:
                logger.error(f"Failed to process embeddings for: {email}, will retry")
                message.nack()

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message {message.message_id}: {e}")
            message.ack()  # Don't retry malformed messages
        except Exception as e:
            logger.error(
                f"Unexpected error processing message {message.message_id}: {e}",
                exc_info=True,
            )
            message.nack()  # Retry on unexpected errors

    def start(self) -> None:
        """Start the streaming pull subscriber (blocking)."""
        logger.info(f"Starting PubSub subscriber on {self._subscription_path}...")

        flow_control = pubsub_v1.types.FlowControl(
            max_messages=configs.PUBSUB_MAX_MESSAGES,
        )

        self._streaming_pull_future = self._subscriber.subscribe(
            self._subscription_path,
            callback=self._handle_message,
            flow_control=flow_control,
        )

        # Handle graceful shutdown
        def _shutdown(signum, frame):
            logger.info("Shutdown signal received, cancelling subscriber...")
            if self._streaming_pull_future:
                self._streaming_pull_future.cancel()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        logger.info("PubSub subscriber is running. Waiting for messages...")
        print("=" * 60)
        print(f"  Face Matching Worker")
        print(f"  Listening on: {self._subscription_path}")
        print(f"  Press Ctrl+C to stop")
        print("=" * 60)

        try:
            self._streaming_pull_future.result()
        except TimeoutError:
            logger.warning("Subscriber timed out, restarting...")
            self._streaming_pull_future.cancel()
            self._streaming_pull_future.result()
        except KeyboardInterrupt:
            logger.info("Subscriber interrupted by user")
            self._streaming_pull_future.cancel()
        except Exception as e:
            logger.error(f"Subscriber error: {e}", exc_info=True)
            self._streaming_pull_future.cancel()

    def stop(self) -> None:
        """Stop the subscriber."""
        if self._streaming_pull_future:
            self._streaming_pull_future.cancel()
            logger.info("PubSub subscriber stopped")
