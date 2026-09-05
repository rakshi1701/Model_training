import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

// Host utilisation, refreshed every 2s by the SystemMonitor backend.
Card {
    id: panel
    property var s: Monitor.stats

    RowLayout {
        Layout.fillWidth: true
        SectionTitle { title: "📊 System Utilization Monitor" }
        Item { Layout.fillWidth: true }
        StatusBadge { text_: "🔴 Live"; tint: Theme.danger }
    }

    // Row 1 — headline numbers
    RowLayout {
        Layout.fillWidth: true
        spacing: 10
        MetricCard {
            label: "CPU (" + (s.cpu_count_physical || 0) + "P/" + (s.cpu_count_logical || 0) + "L"
                   + (s.cpu_freq ? " @ " + Math.round(s.cpu_freq) + " MHz" : "") + ")"
            value: (s.cpu_percent || 0).toFixed(1) + "%"
            valueColor: Theme.colorFor(s.cpu_percent || 0)
        }
        MetricCard {
            label: "RAM Usage"
            value: (s.ram_used_gb || 0).toFixed(1) + " / " + (s.ram_total_gb || 0).toFixed(1) + " GB"
            delta: (s.ram_percent || 0).toFixed(1) + "% used"
            inverse: true
            valueColor: Theme.colorFor(s.ram_percent || 0)
        }
        MetricCard {
            label: "Swap"
            value: (s.swap_used_gb || 0).toFixed(1) + " / " + (s.swap_total_gb || 0).toFixed(1) + " GB"
            delta: s.swap_total_gb > 0 ? (s.swap_percent || 0).toFixed(1) + "% used" : "N/A"
            inverse: true
        }
        MetricCard {
            label: "Disk (/)"
            value: Math.round(s.disk_used_gb || 0) + " / " + Math.round(s.disk_total_gb || 0) + " GB"
            delta: (s.disk_percent || 0).toFixed(1) + "% used"
            inverse: true
            valueColor: Theme.colorFor(s.disk_percent || 0)
        }
    }

    // Row 2 — per-core bars and memory allocation
    RowLayout {
        Layout.fillWidth: true
        spacing: 12

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 262
            radius: 10
            color: Qt.rgba(0.06, 0.09, 0.16, 0.6)
            border.color: Theme.border
            border.width: 1
            clip: true
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 6
                Text {
                    text: "🧠 CPU Core Utilization"
                    color: Theme.accent
                    font.pixelSize: Theme.fsSmall
                    font.bold: true
                }
                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    ColumnLayout {
                        width: parent.parent.width
                        spacing: 3
                        Repeater {
                            model: s.cpu_per_core || []
                            delegate: BarMeter {
                                required property var modelData
                                required property int index
                                labelWidth: 34
                                barHeight: 13
                                label: "C" + (index < 10 ? "0" + index : index)
                                value: (modelData || 0) / 100
                                valueText: Math.round(modelData || 0) + "%"
                                barColor: Theme.colorFor(modelData || 0)
                            }
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 262
            radius: 10
            color: Qt.rgba(0.06, 0.09, 0.16, 0.6)
            border.color: Theme.border
            border.width: 1
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 10
                Text {
                    text: "💾 Memory Allocation"
                    color: Theme.accent
                    font.pixelSize: Theme.fsSmall
                    font.bold: true
                }
                BarMeter {
                    label: "RAM"
                    barHeight: 20
                    value: (s.ram_percent || 0) / 100
                    valueText: (s.ram_used_gb || 0).toFixed(1) + " GB / "
                               + (s.ram_total_gb || 0).toFixed(1) + " GB ("
                               + (s.ram_percent || 0).toFixed(1) + "%)"
                    barColor: Theme.colorFor(s.ram_percent || 0)
                }
                RowLayout {
                    Layout.fillWidth: true
                    Text {
                        text: "Available"
                        color: Theme.textMuted
                        font.pixelSize: Theme.fsSmall
                        Layout.fillWidth: true
                    }
                    Text {
                        text: (s.ram_available_gb || 0).toFixed(1) + " GB free"
                        color: Theme.green
                        font.pixelSize: Theme.fsSmall
                        font.bold: true
                    }
                }
                BarMeter {
                    visible: (s.swap_total_gb || 0) > 0
                    label: "Swap"
                    barHeight: 14
                    value: (s.swap_percent || 0) / 100
                    valueText: (s.swap_used_gb || 0).toFixed(1) + " / "
                               + (s.swap_total_gb || 0).toFixed(1) + " GB"
                    barColor: Theme.colorFor(s.swap_percent || 0)
                }

                Rectangle {
                    Layout.fillWidth: true
                    visible: !!s.train_proc
                    implicitHeight: procCol.implicitHeight + 18
                    radius: 8
                    color: Qt.rgba(0, 0.9, 0.64, 0.08)
                    border.color: Qt.rgba(0, 0.9, 0.64, 0.22)
                    border.width: 1
                    ColumnLayout {
                        id: procCol
                        anchors.fill: parent
                        anchors.margins: 9
                        spacing: 3
                        Text {
                            text: "🏋️ Training Process (PID " + (s.train_proc ? s.train_proc.pid : "—") + ")"
                            color: Theme.success
                            font.pixelSize: Theme.fsSmall
                            font.bold: true
                        }
                        Repeater {
                            model: s.train_proc ? [
                                { k: "Resident Memory", v: s.train_proc.rss_gb.toFixed(2) + " GB" },
                                { k: "CPU Usage", v: s.train_proc.cpu_percent.toFixed(1) + "%" },
                                { k: "Threads", v: String(s.train_proc.num_threads) }
                            ] : []
                            delegate: RowLayout {
                                required property var modelData
                                Layout.fillWidth: true
                                Text {
                                    text: modelData.k
                                    color: Theme.textMuted
                                    font.pixelSize: Theme.fsSmall
                                    Layout.fillWidth: true
                                }
                                Text {
                                    text: modelData.v
                                    color: Theme.textDim
                                    font.pixelSize: Theme.fsSmall
                                    font.bold: true
                                }
                            }
                        }
                    }
                }
                Item { Layout.fillHeight: true }
            }
        }
    }

    // Row 3 — GPU cards
    RowLayout {
        Layout.fillWidth: true
        visible: Monitor.hasGpu
        spacing: 12
        Repeater {
            model: s.gpus || []
            delegate: Rectangle {
                required property var modelData
                Layout.fillWidth: true
                implicitHeight: gpuCol.implicitHeight + 24
                radius: 10
                color: Qt.rgba(0.06, 0.09, 0.16, 0.7)
                border.color: Qt.rgba(0.22, 0.74, 0.97, 0.22)
                border.width: 1
                ColumnLayout {
                    id: gpuCol
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 7
                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            text: "🎮 GPU " + modelData.index
                            color: Theme.accent
                            font.pixelSize: Theme.fsBody
                            font.bold: true
                        }
                        Item { Layout.fillWidth: true }
                        Rectangle {
                            implicitHeight: 20
                            implicitWidth: gname.implicitWidth + 16
                            radius: 10
                            color: Qt.rgba(1, 1, 1, 0.06)
                            Text {
                                id: gname
                                anchors.centerIn: parent
                                text: modelData.name
                                color: Theme.textMuted
                                font.pixelSize: Theme.fsTiny
                            }
                        }
                    }
                    BarMeter {
                        label: "GPU Compute"
                        value: (modelData.util_percent || 0) / 100
                        valueText: Math.round(modelData.util_percent || 0) + "%"
                        barColor: (modelData.util_percent || 0) > 80 ? Theme.success
                                : (modelData.util_percent || 0) > 40 ? Theme.accent : Theme.textMuted
                        barHeight: 15
                    }
                    BarMeter {
                        label: "VRAM"
                        value: (modelData.mem_percent || 0) / 100
                        valueText: Math.round(modelData.mem_used_mb || 0) + " / "
                                   + Math.round(modelData.mem_total_mb || 0) + " MB ("
                                   + (modelData.mem_percent || 0).toFixed(1) + "%)"
                        barColor: Theme.colorFor(modelData.mem_percent || 0)
                        barHeight: 15
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            text: "🌡️ Temperature"
                            color: Theme.textMuted
                            font.pixelSize: Theme.fsSmall
                            Layout.fillWidth: true
                        }
                        Text {
                            text: modelData.temp_c ? Math.round(modelData.temp_c) + "°C" : "N/A"
                            color: (modelData.temp_c || 0) > 80 ? Theme.danger
                                 : (modelData.temp_c || 0) > 65 ? Theme.warn : Theme.green
                            font.pixelSize: Theme.fsSmall
                            font.bold: true
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        visible: (modelData.power_w || 0) > 0
                        Text {
                            text: "⚡ Power"
                            color: Theme.textMuted
                            font.pixelSize: Theme.fsSmall
                            Layout.fillWidth: true
                        }
                        Text {
                            text: Math.round(modelData.power_w || 0) + "W / "
                                  + Math.round(modelData.power_limit_w || 0) + "W"
                            color: Theme.textDim
                            font.pixelSize: Theme.fsSmall
                            font.bold: true
                        }
                    }
                }
            }
        }
    }
}
