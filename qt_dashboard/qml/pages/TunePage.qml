import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

// Tab 6 — hyperparameter optimization with `yolo tune`.
Item {
    id: page

    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: page.width - 28
            x: 14
            y: 12
            spacing: 12

            SectionTitle {
                title: "🎛️ Automated Hyperparameter Optimization Studio"
                subtitle: "Search learning rates, momentum, loss gains and augmentation settings with Ultralytics' genetic/Optuna tuner."
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignTop
                spacing: 12

                Card {
                    Layout.preferredWidth: page.width * 0.34
                    Layout.fillWidth: false
                    Layout.alignment: Qt.AlignTop
                    SectionTitle { title: "1. Tuning Search Settings" }

                    Field {
                        label: "Base Checkpoint"
                        StudioCombo {
                            id: baseCombo
                            model: App.models
                            currentIndex: App.defaultModelIndex
                        }
                    }
                    Field {
                        label: "Search Iterations / Trials"
                        StudioSpin { id: trialsBox; realFrom: 2; realTo: 300; value: 15 }
                    }
                    Field {
                        label: "Epochs per Trial"
                        StudioSpin { id: tEpochsBox; realFrom: 1; realTo: 100; value: 10 }
                    }
                    Field {
                        label: "Optimizer"
                        StudioCombo { id: tOptCombo; model: ["auto", "AdamW", "SGD"] }
                    }
                    Field {
                        label: "Tuning Experiment Name"
                        StudioText { id: tNameField; text: Tuner.runName }
                    }
                    Text {
                        text: "Compute: " + (App.cudaAvailable ? "GPU 0" : "CPU")
                        color: Theme.textMuted
                        font.pixelSize: Theme.fsTiny
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        StudioButton {
                            text: "🚀 Start Tuning"
                            kind: "primary"
                            Layout.fillWidth: true
                            enabled: App.hasDataset && !Tuner.active
                            onClicked: Tuner.start({
                                "model": baseCombo.currentText,
                                "data": App.dataYamlPath,
                                "epochs": tEpochsBox.value,
                                "iterations": trialsBox.value,
                                "optimizer": tOptCombo.currentText,
                                "device": App.cudaAvailable ? "0" : "cpu",
                                "name": tNameField.text
                            })
                        }
                        StudioButton {
                            text: "🛑 Stop"
                            kind: "danger"
                            Layout.fillWidth: true
                            enabled: Tuner.active
                            onClicked: Tuner.stop()
                        }
                    }
                    InfoNote {
                        text_: App.hasDataset ? "" : "Select a dataset in the sidebar before tuning."
                        kind: "warn"
                    }
                    StatusBadge {
                        text_: Tuner.active ? "🟢 Tuning: " + Tuner.runName : "⚪ Tuner Idle"
                        tint: Tuner.active ? Theme.success : Theme.textMuted
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignTop
                    spacing: 12

                    Card {
                        SectionTitle { title: "2. Live Tuning Log" }
                        LogView { text_: Tuner.logText; implicitHeight: 300 }
                    }

                    Card {
                        visible: Tuner.bestParams.length > 0
                        SectionTitle {
                            title: "🏆 Optimal Hyperparameters Discovered"
                            subtitle: Tuner.bestSource
                        }
                        DataTable {
                            columns: [{ title: "Parameter", key: "key", width: 220 },
                                      { title: "Value", key: "value" }]
                            rows: Tuner.bestParams
                            maxHeight: 300
                        }
                        StudioButton {
                            text: "✨ Save & Apply to Training Settings"
                            kind: "primary"
                            Layout.fillWidth: true
                            onClicked: {
                                App.setTunedParams(Tuner.bestParamsMap)
                                Notifier.notify("Tuned hyperparameters transferred to the Training tab.", "success")
                            }
                        }
                    }
                }
            }
            Item { Layout.preferredHeight: 10 }
        }
    }
}
