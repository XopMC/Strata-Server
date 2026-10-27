#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Actual N1 new-owner diagnostic snapshot. No Engine cast or ownership token.
   Registry is weak DIAGNOSTIC metadata only; actual Handle/Lease/Task retain Owner.
   rc0 exact64words,1badargs,2absent/closing/closed,3multiple,4busy,5exception.
   Error returns preserve allzero output. Read afterPF/afterwarmup/afterGEN while
   target idle, before terminalclose; afterclose rc2. Cumulative counters, noreset.
   Legacy Receipt370 remains EXACT672B and gap(max2) semantics are not reused. */
int ds469_fixed_plan_info(uint64_t*out,int words)
#ifdef __cplusplus
 noexcept
#endif
;
#ifdef __cplusplus
}
namespace ds469 {
enum InfoIndex {
 Version,Words,Mode,LiveOwner,Generation,FormatEpoch,Catalog,Physical,Active,Spares,
 PermanentStatics,WorkerCPU,WorkerInitialized,Armed,TrainingForwards,BeforeCalls,
 AfterCalls,LayerCursor,PartialForward,CurrentM,TrainingRows,Frozen,PlanBuilt,
 PlanMoves,QueuedMoves,Started,Ready,Published,Cancelled,Pending,MaxPending,
 BackpressureParks,ProtectedReady,NewRoutesUsed,OldRoutesUsed,ComputeBegins,
 ComputeUploadPending,H2DBytes,InverseReadBytes,InverseWriteBytes,ValidationNs,
 CopyNs,PlannerNs,PlannerThreadNs,StageThreadNs,WorkerParked,WorkerJoined,
 UploadDrained,ComputeDrained,LivePinnedBytes,LiveDeviceBytes,DeclaredPeakBytes,
 Errors,Poisoned,ActiveTicket,Borrowed,Closed,HookRefs,StartupInverseCalls,
 StartupInverseBytes,StartupJobs,ObserverErrors,Reserved62,Reserved63,InfoWords
};
static_assert(InfoWords==64);
}
#endif
/* Immutable flags, both matched512 OFF andON:
 DS1017_FIXED_PLAN_UPLOAD_V465=1 (strict252cohort/reserved253),
 DS1017_FIXED_PLAN_REFRESH_V469=0|1 (actual OFF/ON mode),
 DS1017_WHOLE_ADAPTIVE_V370=0 (legacy repeatedpolicy disabled).
 Existing typed387Handle ARM is reused AFTER warmeddriverentry. Driver passes
 gap_control386=None; no old387gap service/copy/replanning. Callback Task254.
 Eventpending counter is a post-enqueue upload-event observation, NOT CTA proof.
 TrainingForwards stops32; it is NOT total generation targetcalls. */
