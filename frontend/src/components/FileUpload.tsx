import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X, CheckCircle, AlertCircle } from 'lucide-react';
import { UploadResponse } from '../types/api';
import ApiService from '../services/api';
import { InlineLoading } from './Loading';
import toast from 'react-hot-toast';

interface FileUploadProps {
  onUploadSuccess: (response: UploadResponse) => void;
  isUploading: boolean;
  setIsUploading: (loading: boolean) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ 
  onUploadSuccess, 
  isUploading, 
  setIsUploading 
}) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<string>('');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
      toast.error('Please upload a CSV file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast.error('File size must be less than 10MB');
      return;
    }

    setUploadedFile(file);
    setIsUploading(true);
    setUploadProgress('Uploading and processing...');

    try {
      const response = await ApiService.uploadFile(file);
      toast.success('File uploaded successfully!');
      onUploadSuccess(response);
      setUploadProgress('Processing complete');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Upload failed';
      toast.error(message);
      setUploadedFile(null);
    } finally {
      setIsUploading(false);
      setUploadProgress('');
    }
  }, [onUploadSuccess, setIsUploading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv']
    },
    multiple: false,
    disabled: isUploading
  });

  const removeFile = () => {
    setUploadedFile(null);
    setUploadProgress('');
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive 
            ? 'border-primary-400 bg-primary-50' 
            : 'border-slate-300 hover:border-primary-400 hover:bg-slate-50'
          }
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center space-y-4">
          <div className={`
            p-4 rounded-full transition-colors duration-200
            ${isDragActive ? 'bg-primary-100' : 'bg-slate-100'}
          `}>
            <Upload className={`
              h-8 w-8 transition-colors duration-200
              ${isDragActive ? 'text-primary-600' : 'text-slate-400'}
            `} />
          </div>
          
          <div>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">
              {isDragActive ? 'Drop your CSV file here' : 'Upload your CSV data'}
            </h3>
            <p className="text-slate-500 mb-2">
              Drag and drop your CSV file here, or click to browse
            </p>
            <p className="text-sm text-slate-400">
              Supports CSV files up to 10MB
            </p>
          </div>
        </div>

        {isUploading && (
          <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center rounded-xl">
            <InlineLoading message={uploadProgress} />
          </div>
        )}
      </div>

      {/* Uploaded File Info */}
      {uploadedFile && !isUploading && (
        <div className="card p-4 border-success-200 bg-success-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-success-100 rounded-lg">
                <CheckCircle className="h-5 w-5 text-success-600" />
              </div>
              <div>
                <h4 className="font-medium text-success-800">{uploadedFile.name}</h4>
                <p className="text-sm text-success-600">
                  {formatFileSize(uploadedFile.size)} • Ready for processing
                </p>
              </div>
            </div>
            <button
              onClick={removeFile}
              className="p-1 text-success-600 hover:text-success-800 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* File Requirements */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="font-medium text-blue-800 mb-2">File Requirements</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• CSV format with headers in the first row</li>
              <li>• Numeric data will be used for training</li>
              <li>• Categorical data will be automatically encoded</li>
              <li>• Missing values will be handled automatically</li>
              <li>• Data will be cleaned and preprocessed for optimal results</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Sample Data Preview */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Expected Data Format</h3>
        <div className="bg-slate-50 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm text-slate-600">
{`feature1,feature2,feature3,target
1.2,3.4,5.6,10.1
2.1,4.2,6.8,12.3
3.5,2.1,7.9,15.2
...`}
          </pre>
        </div>
        <p className="text-sm text-slate-500 mt-2">
          The last column will be used as the target variable for training
        </p>
      </div>
    </div>
  );
};

export default FileUpload;