#include "SileroVADSubsystem.h"
#include "Sound/SoundWave.h"
#include "AudioDevice.h"
#include "NNE.h"
#include "NNERuntimeRunSync.h"
#include "DSP/BufferVectorOperations.h"


void USileroVADSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	TObjectPtr<UNNEModelData> ModelData = LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/SileroVAD/ThirdParty/ONNX/silero_vad.onnx"));
	TWeakInterfacePtr<INNERuntimeGPU> Runtime = UE::NNE::GetRuntime<INNERuntimeGPU>(FString("NNERuntimeORTGpu"));

	ModelInstance = Runtime->CreateModelGPU(ModelData)->CreateModelInstanceGPU();

	if (ModelInstance)
	{
		SetupModel();
	}
}

void USileroVADSubsystem::SetupModel()
{
	HiddenH.Init(0.f, 2 * 1 * 64); // Shape [2,1,64]
	HiddenC.Init(0.f, 2 * 1 * 64);
}

TArray<float> USileroVADSubsystem::ConvertPCM16ToFloat(const TArray<int16>& PCMData)
{
	TArray<float> Result;
	Result.Reserve(PCMData.Num());

	for (int16 Sample : PCMData)
		Result.Add(static_cast<float>(Sample) / 32768.f);

	return Result;
}

void USileroVADSubsystem::AnalyzeSoundWave(USoundWave* SoundWave, float Threshold)
{
	if (!SoundWave || !ModelInstance.IsValid())
		return;

	TArray<uint8> RawData;
	RawData.Append(SoundWave->GetResourceData(), SoundWave->GetResourceSize());

	const int16* PCMPtr = reinterpret_cast<const int16*>(RawData.GetData());
	int32 NumSamples = RawData.Num() / sizeof(int16);
	TArray<int16> PCMData(PCMPtr, NumSamples);

	TArray<float> AudioFloat = ConvertPCM16ToFloat(PCMData);

	const int32 ChunkSize = 1536;
	TArray<float> InputBuffer;
	InputBuffer.SetNumZeroed(ChunkSize);

	for (int32 i = 0; i < AudioFloat.Num(); i += ChunkSize)
	{
		int32 CopyLen = FMath::Min(ChunkSize, AudioFloat.Num() - i);
		FMemory::Memcpy(InputBuffer.GetData(), AudioFloat.GetData() + i, CopyLen * sizeof(float));
		if (CopyLen < ChunkSize)
			FMemory::Memzero(InputBuffer.GetData() + CopyLen, (ChunkSize - CopyLen) * sizeof(float));

		// Build input bindings
		TArrayView<const UE::NNE::FTensorBindingCPU> InputBindings;
		TArrayView<UE::NNE::FTensorBindingCPU> OutputBindings;

		// Prepare input tensors (by name and index order!)
		TArray<UE::NNE::FTensorBindingCPU> Inputs;
		TArray<UE::NNE::FTensorBindingCPU> Outputs;

		Inputs.Add({ InputBuffer.GetData(), static_cast<uint64>(ChunkSize * sizeof(float)) });
		Inputs.Add({ HiddenH.GetData(), static_cast<uint64>(HiddenH.Num() * sizeof(float)) });
		Inputs.Add({ HiddenC.GetData(), static_cast<uint64>(HiddenC.Num() * sizeof(float)) });

		// Prepare outputs: prob, hn, cn
		float OutputProb = 0.f;
		Outputs.Add({ &OutputProb, sizeof(float) });

		TArray<float> NewH;
		NewH.SetNumUninitialized(HiddenH.Num());

		TArray<float> NewC;
		NewC.SetNumUninitialized(HiddenC.Num());

		Outputs.Add({ NewH.GetData(), static_cast<uint64>(NewH.Num() * sizeof(float)) });
		Outputs.Add({ NewC.GetData(), static_cast<uint64>(NewC.Num() * sizeof(float)) });

		// Execute inference
		if (UE::NNE::IModelInstanceGPU* ModelGPU = static_cast<UE::NNE::IModelInstanceGPU*>(ModelInstance.Get()))
		{
			ModelGPU->RunSync(Inputs, Outputs);

			// Update LSTM state
			HiddenH = NewH;
			HiddenC = NewC;

			bool bSpeaking = OutputProb >= Threshold;

			if (bSpeaking && !bWasSpeaking)
				OnVoiceStarted.Broadcast();
			else if (!bSpeaking && bWasSpeaking)
				OnVoiceStopped.Broadcast();

			bWasSpeaking = bSpeaking;
		}
	}
}
