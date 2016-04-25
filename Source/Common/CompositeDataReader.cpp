//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

#define DATAREADER_LOCAL

#include "CompositeDataReader.h"
#include "Bundler.h"

#include "BlockRandomizer.h"
#include "NoRandomizer.h"
#include "FramePacker.h"
#include "SequencePacker.h"
#include "HeapMemoryProvider.h"
#include "DataDeserializer.h"
#include "Transformer.h"
#include "Reader.h"
#include "Packer.h"
#include "CorpusDescriptor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

CompositeDataReader::CompositeDataReader(const std::string& precision) : m_layout(make_shared<MBLayout>()),
    m_precision(precision),
    m_corpus(std::make_shared<CorpusDescriptor>())
{
}

void CompositeDataReader::Init(const ConfigParameters& config)
{
    intargvector numberOfuttsPerMinibatchForAllEpochs =
        config(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int> { 1 })));

    bool prefetch = config(L"prefetch", true);

    m_provider = std::make_shared<HeapMemoryProvider>();

    // if prefetch - launching asynchronously,
    // otherwise deferring - synchronous execution during .get() call
    m_launchType = prefetch ? launch::async : launch::deferred;

    auto numSeqsPerMBForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;
    m_layout->Init(numSeqsPerMBForAllEpochs[0], 0);

    // Check mode.
    m_frameMode = config(L"frameMode", true);
    m_truncated = config(L"truncated", false);

    // Whether we need to check data between different deserializers.
    bool cleanse = config(L"checkData", false);

    // Creating deserializers.
    CreateDeserializers(config);

    // Bundling deserializers together.
    // TODO: Add transformers in between.
    auto bundler = std::make_shared<Bundler>(config, m_deserializers[0], m_deserializers, cleanse);

    int verbosity = config(L"verbosity", 2);

    // Pick up the randomizer.
    bool randomize = config(L"randomize", false);
    if (randomize)
    {
        // By default randomizing the whole data set.
        size_t randomizationWindow = config(L"randomizationWindow", requestDataSize);
        m_randomizer = std::make_shared<BlockRandomizer>(verbosity, randomizationWindow, bundler, BlockRandomizer::DecimationMode::chunk, true);
    }
    else
    {
        m_randomizer = std::make_shared<NoRandomizer>(bundler);
    }

    // Create output stream descriptions - where to get those? from config? what if it is not the same as network expects?
    for (const auto& streamDescription : bundler->GetStreamDescriptions())
    {
        StreamDescriptionPtr stream = std::make_shared<StreamDescription>(*streamDescription);
        stream->m_storageType = StorageType::dense;
        m_streams.push_back(stream);
        m_nameToStreamId.insert(std::make_pair(streamDescription->m_name, streamDescription->m_id));
    }
}

void CompositeDataReader::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
}

void CompositeDataReader::StartDistributedMinibatchLoop(
    size_t requestedMBSize,
    size_t epoch,
    size_t subsetNum,
    size_t numSubsets,
    size_t requestedEpochSamples /*= requestDataSize*/)
{
    EpochConfiguration config;
    config.m_workerRank = subsetNum;
    config.m_numberOfWorkers = numSubsets;
    config.m_minibatchSizeInSamples = requestedMBSize;
    config.m_totalEpochSizeInSamples = requestedEpochSamples;
    config.m_epochIndex = epoch;

    m_endOfEpoch = false;

    // Make sure there are no outstanding reads.
    if (m_prefetchTask.valid())
    {
        m_prefetchTask.wait();
    }

    // Nothing is running, let's reconfigure the packer according to the new epoch.
    StartEpoch(config);

    // Ok, start reading in sync or async manner.
    m_prefetchTask = std::async(m_launchType, [this]()
    {
        return m_packer->ReadMinibatch();
    });
}

bool CompositeDataReader::GetMinibatch(StreamMinibatchInputs& matrices)
{
    if (m_endOfEpoch)
    {
        return false;
    }

    // Check that all matrices have the same device id.
    // If not we should inject the IMemoryProvider per stream.
    int deviceId = matrices.begin()->second->GetDeviceId();
    for (auto mx : matrices)
    {
        if (mx.second->GetDeviceId() != deviceId)
        {
            assert(false);
        }
    }

    assert(m_prefetchTask.valid());

    Minibatch minibatch = m_prefetchTask.get();
    if (minibatch.m_endOfEpoch)
    {
        m_endOfEpoch = true;
        if (minibatch.m_data.empty())
        {
            return false;
        }
    }

    if (!minibatch.m_data.empty())
    {
        // TODO: Use alternating pinned buffer in the packer, do not copy anything, but pack into the pinned memory.
        // Copy returned minibatch to the matrices.
        for (const auto& mx : matrices)
        {
            assert(m_nameToStreamId.find(mx.first) != m_nameToStreamId.end());
            size_t streamId = m_nameToStreamId[mx.first];

            const auto& stream = minibatch.m_data[streamId];
            m_layout->CopyFrom(stream->m_layout);

            size_t columnNumber = m_layout->GetNumCols();
            size_t rowNumber = m_streams[streamId]->m_sampleLayout->GetNumElements();

            if (m_precision == "float")
            {
                auto* data = reinterpret_cast<const float*>(stream->m_data);
                matrices.GetInputMatrix<float>(mx.first).SetValue(rowNumber, columnNumber, mx.second->GetDeviceId(), const_cast<float*>(data), matrixFlagNormal);
            }
            else
            {
                assert(m_precision == "double");
                auto* data = reinterpret_cast<const double*>(stream->m_data);
                matrices.GetInputMatrix<double>(mx.first).SetValue(rowNumber, columnNumber, mx.second->GetDeviceId(), const_cast<double*>(data), matrixFlagNormal);
            }
        }
    }

    m_prefetchTask = std::async(m_launchType, [this]()
    {
        return m_packer->ReadMinibatch();
    });

    return !minibatch.m_data.empty();
}

bool CompositeDataReader::DataEnd()
{
    // Note: Return value never used.
    return false;
}

void CompositeDataReader::CopyMBLayoutTo(MBLayoutPtr layout)
{
    layout->CopyFrom(m_layout);
}

size_t CompositeDataReader::GetNumParallelSequences()
{
    return m_layout->GetNumParallelSequences();
}

void CompositeDataReader::CreateDeserializers(const ConfigParameters& readerConfig)
{
    argvector<ConfigValue> deserializerConfigs =
        readerConfig(L"deserializers", ConfigParameters::Array(argvector<ConfigValue>(vector<ConfigValue> {})));

    assert(m_deserializers.empty());
    bool primary = true;  // CUrrently, the first deserializer becomes primary - it drives chunking.
    for (size_t i = 0; i < deserializerConfigs.size(); ++i)
    {
        // TODO: Should go away in the future. Framing can be done on top of deserializer.
        ConfigParameters p = deserializerConfigs[i];
        p.Insert("frameMode", m_frameMode ? "true" : "false");
        p.Insert("precision", m_precision);

        IDataDeserializerPtr d = CreateDeserializer(p, primary);
        primary = false;
        m_deserializers.push_back(d);
    }
}

IDataDeserializerPtr CompositeDataReader::CreateDeserializer(const ConfigParameters& deserializerConfig, bool primary)
{
    typedef bool(*CreateDeserializerFactory) (IDataDeserializer** d, const std::wstring& type, const ConfigParameters& cfg, CorpusDescriptorPtr corpus, bool primary);

    std::string deserializerModule = deserializerConfig("module");
    CreateDeserializerFactory f = (CreateDeserializerFactory)Plugin::Load(deserializerModule, "CreateDeserializer");

    std::wstring deserializerType = deserializerConfig("type");
    IDataDeserializer* d;
    if (!f(&d, deserializerType, deserializerConfig, m_corpus, primary))
    {
        RuntimeError("Cannot create deserializer. Please check module and type in the configuration.");
    }

    assert(d != nullptr);
    return IDataDeserializerPtr(d);
}

void CompositeDataReader::StartEpoch(const EpochConfiguration& config)
{
    if (config.m_totalEpochSizeInSamples <= 0)
    {
        RuntimeError("Unsupported minibatch size '%d'.", (int)config.m_totalEpochSizeInSamples);
    }

    m_randomizer->StartEpoch(config);

    // TODO: should we unify sample and sequence mode packers into a single one.
    // TODO: functionally they are the same, the only difference is how we handle
    // TODO: MBlayout and what is the perf hit for iterating/copying sequences.
    // TODO: Should do more perf tests before unifying these two.
    if (m_frameMode)
    {
        m_packer = std::make_shared<FramePacker>(
            m_provider,
            m_randomizer,
            config.m_minibatchSizeInSamples,
            m_streams);
    }
    else
    {
        m_packer = std::make_shared<SequencePacker>(
            m_provider,
            m_randomizer,
            config.m_minibatchSizeInSamples,
            m_streams);
    }
}

}}}

