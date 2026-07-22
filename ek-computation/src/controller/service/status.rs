//! Map Controller runtime-state failures to gRPC status codes.

use tonic::Status;

use crate::controller::runtime_state::ControllerStateError;

pub(super) fn state_status(error: ControllerStateError) -> Status {
    match error {
        ControllerStateError::InvalidRegistration(_)
        | ControllerStateError::InvalidHeartbeatState
        | ControllerStateError::InvalidExpertState
        | ControllerStateError::FutureTopologyVersion { .. } => {
            Status::invalid_argument(error.to_string())
        }
        ControllerStateError::UnknownWorker
        | ControllerStateError::ReplacedWorker
        | ControllerStateError::StaleHeartbeatStream
        | ControllerStateError::HeartbeatSequenceRollback { .. }
        | ControllerStateError::PlacementTooLarge { .. }
        | ControllerStateError::PlacementGenerationMismatch { .. }
        | ControllerStateError::ReportSequenceRollback { .. }
        | ControllerStateError::ConflictingReportSequence(_)
        | ControllerStateError::UnknownDrain(_) => Status::failed_precondition(error.to_string()),
        ControllerStateError::Persistence(_) => Status::internal(error.to_string()),
    }
}
