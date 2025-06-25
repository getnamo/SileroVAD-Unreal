#pragma once

#include "CoreMinimal.h"
#include "Subsystems/EngineSubsystem.h"
#include "NNEModelData.h"
#include "NNERuntime.h"
#include "NNERuntimeFormat.h"
#include "NNERuntimeGPU.h"
#include "SileroVADSubsystem.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FVoiceActivityEvent);

UCLASS()
class SILEROVAD_API USileroVADSubsystem : public UEngineSubsystem
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintAssignable, Category = "Silero VAD")
	FVoiceActivityEvent OnVoiceStarted;

	UPROPERTY(BlueprintAssignable, Category = "Silero VAD")
	FVoiceActivityEvent OnVoiceStopped;

	UFUNCTION(BlueprintCallable, Category = "Silero VAD")
	void AnalyzeSoundWave(USoundWave* SoundWave, float Threshold = 0.5f);

protected:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;

private:
	TSharedPtr<UE::NNE::IModelInstanceGPU> ModelInstance;

	TArray<float> HiddenH;
	TArray<float> HiddenC;

	bool bWasSpeaking = false;

	void SetupModel();
	TArray<float> ConvertPCM16ToFloat(const TArray<int16>& PCMData);
};
