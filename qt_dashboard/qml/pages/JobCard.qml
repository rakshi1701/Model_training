import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

// One self-contained vertical card per Kaggle job — status, facts, model
// downloads, actions and an expandable log + GPU telemetry panel.
Card {
    id: card
    property var job: ({})
    property bool showDetail: false
    property bool confirmRemove: false
    property var progress: ({})

    signal saveWeight(string path)

    Connections {
        target: Kaggle
        function onProgressChanged(ref) {
            if (ref === card.job.ref) {
                card.progress = Kaggle.progressFor(ref)
                card.showDetail = true
            }
        }
    }

    // ---- header ----
    RowLayout {
        Layout.fillWidth: true
        spacing: 10
        StatusBadge {
            text_: job.catIcon + " " + job.catName
            tint: job.catColour
        }
        Text {
            text: job.shortName
            color: Theme.text
            font.pixelSize: Theme.fsTitle
            font.bold: true
            elide: Text.ElideMiddle
            Layout.fillWidth: true
        }
        Text {
            text: "· " + job.whenText
            color: Theme.textMuted
            font.pixelSize: Theme.fsSmall
        }
        StudioButton {
            text: "🔗 Kaggle Console"
            kind: "ghost"
            onClicked: App.openUrl(Kaggle.kernelUrl(job.ref))
        }
    }

    // ---- facts ----
    RowLayout {
        Layout.fillWidth: true
        spacing: 10
        Repeater {
            model: [{ k: "Model", v: String(job.model_name) },
                    { k: "Epochs", v: String(job.epochs) },
                    { k: "Dataset", v: String(job.datasetShort) },
                    { k: "Runtime", v: String(job.runtimeText) }]
            delegate: ColumnLayout {
                required property var modelData
                Layout.fillWidth: true
                spacing: 1
                Text {
                    text: modelData.k
                    color: Theme.textMuted
                    font.pixelSize: Theme.fsTiny
                }
                Text {
                    text: modelData.v
                    color: Theme.textDim
                    font.pixelSize: Theme.fsSmall
                    font.family: Theme.mono
                    elide: Text.ElideMiddle
                    Layout.fillWidth: true
                }
            }
        }
    }

    InfoNote {
        text_: (job.category === "terminated" || job.category === "cancelled")
               ? job.reason + "." : ""
        kind: "warn"
    }
    InfoNote {
        text_: job.category === "failed" ? job.reason + "." : ""
        kind: "error"
    }
    InfoNote {
        text_: job.category === "unresolved" ? job.reason + "." : ""
        kind: "info"
    }
    InfoNote {
        text_: job.failureMessage ? "Kernel failure: " + job.failureMessage : ""
        kind: "error"
    }
    Text {
        visible: !!job.discovered
        text: "📡 Found on your Kaggle account — not dispatched from here, so its hyperparameters aren't known locally."
        color: Theme.textMuted
        font.pixelSize: Theme.fsTiny
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
    Text {
        visible: job.resumed_from !== ""
        text: "♻️ Continues " + job.resumed_from
        color: Theme.textMuted
        font.pixelSize: Theme.fsTiny
        Layout.fillWidth: true
    }

    // ---- model downloads ----
    ColumnLayout {
        Layout.fillWidth: true
        visible: job.category === "successful" || job.isPartial
        spacing: 6
        Text {
            text: job.category === "successful" ? "🎯 Trained model" : "🧩 Partial checkpoint"
            color: Theme.textDim
            font.pixelSize: Theme.fsSmall
            font.bold: true
        }
        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            visible: job.weights.length > 0
            Repeater {
                model: job.weights
                delegate: StudioButton {
                    required property var modelData
                    text: "💾 Save " + modelData.name
                          + (card.job.isPartial ? " (partial)" : "")
                          + " · " + modelData.size
                    Layout.fillWidth: true
                    onClicked: card.saveWeight(modelData.path)
                }
            }
            StudioButton {
                text: "📂"
                kind: "ghost"
                Layout.preferredWidth: 40
                onClicked: App.revealPath(job.weightsDir)
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            visible: job.weights.length === 0
            spacing: 4
            Text {
                text: job.category === "successful"
                      ? "Weights are still on Kaggle — fetch them to enable the download."
                      : "Not a finished model, but it can seed a resume job."
                color: Theme.textMuted
                font.pixelSize: Theme.fsTiny
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
            StudioButton {
                text: job.category === "successful"
                      ? "⬇ Fetch model (.pt) from Kaggle"
                      : "⬇ Fetch partial checkpoint (.pt)"
                Layout.fillWidth: true
                onClicked: Kaggle.fetchWeights(job.ref)
            }
        }
        Text {
            visible: job.weights.length > 0
            text: "Saved in " + job.weightsDir + " — also available in Inference and Export Studio."
            color: Theme.textMuted
            font.pixelSize: Theme.fsTiny
            elide: Text.ElideMiddle
            Layout.fillWidth: true
        }
    }

    // ---- actions ----
    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        StudioButton {
            text: "🛑 Stop this job"
            kind: "primary"
            visible: job.canStop
            Layout.fillWidth: true
            onClicked: Kaggle.stopJob(job.ref)
        }
        StudioButton {
            text: "📥 Ingest full run"
            visible: job.canIngest
            Layout.fillWidth: true
            onClicked: Kaggle.ingestRun(job.ref)
        }
        StudioButton {
            text: job.logLabel
            Layout.fillWidth: true
            onClicked: {
                if (Object.keys(card.progress).length > 0) card.showDetail = !card.showDetail
                else Kaggle.loadProgress(job.ref)
            }
        }
        StudioButton {
            text: card.confirmRemove ? "🗑 Press again to remove" : "🗑 Remove"
            kind: card.confirmRemove ? "danger" : "secondary"
            Layout.fillWidth: true
            onClicked: {
                if (job.canStop && !card.confirmRemove) {
                    card.confirmRemove = true
                    Notifier.notify("This job still looks active on Kaggle. Removing it only stops "
                                    + "tracking — the kernel keeps running and keeps using your GPU "
                                    + "quota. Press again to remove.", "warn")
                } else {
                    Kaggle.removeJob(job.ref)
                }
            }
        }
    }

    // ---- progress / telemetry ----
    ColumnLayout {
        Layout.fillWidth: true
        visible: card.showDetail && Object.keys(card.progress).length > 0
        spacing: 8

        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: Theme.border }

        ProgressBarLine {
            visible: (card.progress.totalEpochs || 0) > 0
            value: card.progress.pct || 0
            caption: "Epoch " + (card.progress.epoch || 0) + " / " + (card.progress.totalEpochs || 0)
                     + " · " + Math.round((card.progress.pct || 0) * 100) + "%"
                     + (card.progress.secPerEpoch ? " · " + card.progress.secPerEpoch + "s/epoch" : "")
                     + (card.progress.etaStr ? " · ETA " + card.progress.etaStr : "")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            visible: !!card.progress.metrics && Object.keys(card.progress.metrics).length > 0
            MetricCard {
                label: "mAP@50"
                valueColor: Theme.success
                value: card.progress.metrics ? (card.progress.metrics.mAP50 || 0).toFixed(3) : "—"
            }
            MetricCard {
                label: "mAP@50-95"
                valueColor: Theme.success
                value: card.progress.metrics ? (card.progress.metrics.mAP50_95 || 0).toFixed(3) : "—"
            }
            MetricCard {
                label: "Precision"
                value: card.progress.metrics ? (card.progress.metrics.precision || 0).toFixed(3) : "—"
            }
            MetricCard {
                label: "Recall"
                value: card.progress.metrics ? (card.progress.metrics.recall || 0).toFixed(3) : "—"
            }
        }

        Text {
            visible: (card.progress.gpuLatest || []).length > 0
            text: "🖥️ Remote GPU utilisation"
            color: Theme.textDim
            font.pixelSize: Theme.fsSmall
            font.bold: true
        }
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Repeater {
                model: card.progress.gpuLatest || []
                delegate: RowLayout {
                    required property var modelData
                    Layout.fillWidth: true
                    spacing: 8
                    MetricCard {
                        label: "GPU " + modelData.index + " Util"
                        value: Math.round(modelData.util || 0) + "%"
                    }
                    MetricCard {
                        label: "VRAM"
                        value: ((modelData.mem_used || 0) / 1024).toFixed(1) + " GB"
                        delta: modelData.mem_total ? "of " + (modelData.mem_total / 1024).toFixed(1) + " GB" : ""
                    }
                    MetricCard {
                        label: "Temp"
                        value: Math.round(modelData.temp || 0) + "°C"
                    }
                    MetricCard {
                        label: "Power"
                        value: Math.round(modelData.power || 0) + " W"
                    }
                }
            }
        }
        Text {
            visible: !!card.progress.gpuSummary && card.progress.gpuSummary.n_samples !== undefined
            text: {
                var s = card.progress.gpuSummary || ({})
                return "Avg util " + (s.avg_util || 0) + "% · peak " + (s.peak_util || 0)
                     + "% · peak VRAM " + ((s.peak_mem_mb || 0) / 1024).toFixed(1) + " GB · "
                     + (s.n_samples || 0) + " samples over "
                     + Math.round((s.sampled_seconds || 0) / 60) + " min"
            }
            color: Theme.textMuted
            font.pixelSize: Theme.fsTiny
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
        }

        MetricChart {
            visible: (card.progress.gpuSeries || []).length > 0
            seriesData: card.progress.gpuSeries || []
            xTitle: "Minutes into job"
            yTitle: "GPU util %"
            decimalsY: 0
            implicitHeight: 240
        }

        InfoNote {
            text_: card.progress.errorLine ? "Likely cause: " + card.progress.errorLine : ""
            kind: "warn"
        }
        LogView {
            visible: !!card.progress.logAvailable
            text_: card.progress.tail || ""
            implicitHeight: 240
        }
        InfoNote {
            text_: card.progress.logAvailable ? "" :
                "Kaggle only serves the run log once the session ends, so epoch detail and GPU "
                + "telemetry appear after the job finishes. The status above is live."
            kind: "info"
        }
    }
}
