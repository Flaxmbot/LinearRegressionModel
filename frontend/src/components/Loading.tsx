import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const Loading: React.FC<LoadingProps> = ({ 
  message = 'Loading...', 
  size = 'md',
  className = '' 
}) => {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8'
  };

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg'
  };

  return (
    <div className={`flex flex-col items-center justify-center space-y-3 ${className}`}>
      <div className="relative">
        <Loader2 className={`${sizeClasses[size]} text-primary-600 animate-spin`} />
        <div className="absolute inset-0 rounded-full border-2 border-primary-200"></div>
      </div>
      <p className={`${textSizeClasses[size]} text-slate-600 font-medium animate-pulse`}>
        {message}
      </p>
    </div>
  );
};

export default Loading;

// Full screen loading component
export const FullScreenLoading: React.FC<{ message?: string }> = ({ 
  message = 'Loading...' 
}) => {
  return (
    <div className="fixed inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl p-8 border border-slate-200">
        <Loading message={message} size="lg" />
      </div>
    </div>
  );
};

// Inline loading component
export const InlineLoading: React.FC<{ message?: string }> = ({ 
  message = 'Processing...' 
}) => {
  return (
    <div className="flex items-center space-x-2 text-slate-600">
      <Loader2 className="h-4 w-4 animate-spin" />
      <span className="text-sm font-medium">{message}</span>
    </div>
  );
};

// Loading spinner for buttons
export const ButtonLoading: React.FC = () => {
  return (
    <Loader2 className="h-4 w-4 animate-spin mr-2" />
  );
};

// Skeleton loading components
export const CardSkeleton: React.FC = () => {
  return (
    <div className="card p-6 animate-pulse">
      <div className="h-6 bg-slate-200 rounded w-3/4 mb-4"></div>
      <div className="space-y-3">
        <div className="h-4 bg-slate-200 rounded"></div>
        <div className="h-4 bg-slate-200 rounded w-5/6"></div>
        <div className="h-4 bg-slate-200 rounded w-4/6"></div>
      </div>
    </div>
  );
};

export const TableSkeleton: React.FC<{ rows?: number; cols?: number }> = ({ 
  rows = 5, 
  cols = 4 
}) => {
  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {Array.from({ length: cols }).map((_, i) => (
          <div key={i} className="h-4 bg-slate-200 rounded animate-pulse"></div>
        ))}
      </div>
      
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div key={rowIndex} className="grid gap-4" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
          {Array.from({ length: cols }).map((_, colIndex) => (
            <div key={colIndex} className="h-4 bg-slate-100 rounded animate-pulse"></div>
          ))}
        </div>
      ))}
    </div>
  );
};