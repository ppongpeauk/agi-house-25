"use client"

import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Calendar } from "@/components/ui/calendar"
import { ArrowRight, AlertCircle, ChevronLeft, ChevronRight, XCircle } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

// Default coordinates for Ethiopia
const DEFAULT_LATITUDE = 9
const DEFAULT_LONGITUDE = 39.5

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export function DiseaseRiskPanel() {
  const [date, setDate] = useState(new Date())
  const [latitude, setLatitude] = useState(DEFAULT_LATITUDE)
  const [longitude, setLongitude] = useState(DEFAULT_LONGITUDE)
  const [selectedDisease, setSelectedDisease] = useState(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ["inference", date, latitude, longitude],
    queryFn: async () => {
      try {
        const response = await fetch(`${API_URL}/inference`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Accept": "application/json",
          },
          body: JSON.stringify({
            latitude,
            longitude,
            timestamp: Math.floor(date.getTime() / 1000),
          }),
        })

        if (!response.ok) {
          throw new Error(`API Error: ${response.status}`)
        }

        return response.json()
      } catch (err) {
        console.error("API Error:", err)
        throw new Error("Failed to fetch disease predictions. Please try again later.")
      }
    },
    retry: 2,
  })

  const getRiskColor = (probability) => {
    if (probability >= 0.5) return "text-red-500"
    if (probability >= 0.3) return "text-yellow-500"
    return "text-emerald-500"
  }

  const getRiskBg = (probability) => {
    if (probability >= 0.5) return "bg-red-500/10"
    if (probability >= 0.3) return "bg-yellow-500/10"
    return "bg-emerald-500/10"
  }

  return (
    <div className="p-6 space-y-6">

      {/* Disease Risks */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-white/70">Disease Risks</h3>
        {isLoading ? (
          <div className="text-sm text-white/50 animate-pulse">Loading predictions...</div>
        ) : error ? (
          <div className="bg-red-500/10 text-red-500 p-3 rounded-lg flex items-center gap-2 text-sm">
            <XCircle className="w-4 h-4" />
            <span>{error.message}</span>
          </div>
        ) : (
          <div className="space-y-2">
            {data?.predictions.map((prediction) => (
              <div
                key={prediction.disease_name}
                className={`flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors ${getRiskBg(prediction.probability)} hover:bg-white/5`}
                onClick={() => setSelectedDisease(prediction)}
              >
                <div className="flex items-center gap-2">
                  <span className={getRiskColor(prediction.probability)}>
                    <AlertCircle className="w-4 h-4" />
                  </span>
                  <span className="text-sm text-white">{prediction.disease_name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium ${getRiskColor(prediction.probability)}`}>
                    {(prediction.probability * 100).toFixed(1)}%
                  </span>
                  {prediction.probability >= 0.5 && (
                    <ArrowRight className="w-4 h-4 text-white/30" />
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <Dialog open={!!selectedDisease} onOpenChange={() => setSelectedDisease(null)}>
        <DialogContent className="bg-zinc-900 border border-white/10 text-white">
          <DialogHeader>
            <DialogTitle className="text-lg font-semibold">
              {selectedDisease?.disease_name}
            </DialogTitle>
          </DialogHeader>
          <div className="mt-4">
            <p className="text-sm text-white/70">
              This is a placeholder for LLM-generated content about {selectedDisease?.disease_name}.
            </p>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
