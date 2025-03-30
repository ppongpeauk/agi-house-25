"use client";

import { useState, useEffect } from "react";
import { ArrowRight, AlertCircle, XCircle, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface Prediction {
  disease_name: string;
  probability: number;
}

interface ResearchData {
  research: string;
  date: string;
}

interface DiseaseRiskPanelProps {
  predictions: Prediction[] | undefined;
  isLoading: boolean;
  error: Error | null;
  location: string;
  timestamp: number;
}

export function DiseaseRiskPanel({
  predictions,
  isLoading,
  error,
  location,
  timestamp,
}: DiseaseRiskPanelProps) {
  const [selectedDisease, setSelectedDisease] = useState<Prediction | null>(
    null
  );
  const [researchData, setResearchData] = useState<ResearchData | null>(null);
  const [isResearchLoading, setIsResearchLoading] = useState(false);
  const [researchError, setResearchError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchResearch = async () => {
      if (!selectedDisease) return;

      setIsResearchLoading(true);
      setResearchError(null);

      try {
        const response = await fetch("/api/research", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            disease: selectedDisease.disease_name,
            location,
            timestamp,
          }),
        });

        if (!response.ok) {
          throw new Error("Failed to fetch research data");
        }

        const data = await response.json();
        setResearchData(data);
      } catch (err) {
        setResearchError(
          err instanceof Error ? err : new Error("An error occurred")
        );
      } finally {
        setIsResearchLoading(false);
      }
    };

    fetchResearch();
  }, [selectedDisease, location, timestamp]);

  const getRiskColor = (probability: number): string => {
    if (probability >= 0.5) return "text-red-500";
    if (probability >= 0.3) return "text-yellow-500";
    return "text-emerald-500";
  };

  const getRiskBg = (probability: number): string => {
    if (probability >= 0.5) return "bg-red-500/10";
    if (probability >= 0.3) return "bg-yellow-500/10";
    return "bg-emerald-500/10";
  };

  return (
    <div className="p-6 space-y-6">
      {/* Disease Risks */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-white/70">Disease Risks</h3>
        {isLoading ? (
          <div className="text-sm text-white/50 animate-pulse">
            Loading predictions...
          </div>
        ) : error ? (
          <div className="bg-red-500/10 text-red-500 p-3 rounded-lg flex items-center gap-2 text-sm">
            <XCircle className="w-4 h-4" />
            <span>
              {error instanceof Error ? error.message : "An error occurred"}
            </span>
          </div>
        ) : (
          <div className="space-y-2">
            {predictions?.map((prediction) => (
              <div
                key={prediction.disease_name}
                className={`flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors ${getRiskBg(
                  prediction.probability
                )} hover:bg-white/5`}
                onClick={() => setSelectedDisease(prediction)}
              >
                <div className="flex items-center gap-2">
                  <span className={getRiskColor(prediction.probability)}>
                    <AlertCircle className="w-4 h-4" />
                  </span>
                  <span className="text-sm text-white">
                    {prediction.disease_name}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={`text-sm font-medium ${getRiskColor(
                      prediction.probability
                    )}`}
                  >
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

      <Dialog
        open={!!selectedDisease}
        onOpenChange={() => {
          setSelectedDisease(null);
          setResearchData(null);
        }}
      >
        <DialogContent className="bg-zinc-900 border border-white/10 text-white">
          <DialogHeader>
            <DialogTitle className="text-lg font-semibold">
              {selectedDisease?.disease_name}
            </DialogTitle>
          </DialogHeader>
          <div className="mt-4">
            {isResearchLoading ? (
              <div className="flex items-center gap-2 text-sm text-white/70">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Loading research data...</span>
              </div>
            ) : researchError ? (
              <div className="text-sm text-red-500">
                {researchError.message}
              </div>
            ) : researchData ? (
              <div className="space-y-4">
                <p className="text-sm text-white/50">
                  Research for {researchData.date}
                </p>
                <div className="prose prose-invert max-w-none">
                  {researchData.research.split("\n").map((paragraph, index) => (
                    <p key={index} className="text-sm text-white/70 mb-2">
                      {paragraph}
                    </p>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
