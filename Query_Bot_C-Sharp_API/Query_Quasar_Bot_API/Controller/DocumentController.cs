using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Azure.Storage.Blobs;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

[Route("api/documents")]
[ApiController]
public class DocumentsController : ControllerBase
{
    private readonly string _uploadFolder = "uploaded-documents"; // Updated to use blob storage
    private readonly BlobServiceClient _blobServiceClient;

    public DocumentsController(IConfiguration configuration)
    {
        // Retrieve the Azure Storage connection string from the configuration
        string storageConnectionString = configuration["AzureConnectionStrings:AzureStorageConnectionString"];

        // Initialize the BlobServiceClient with the retrieved connection string
        _blobServiceClient = new BlobServiceClient(storageConnectionString);
    }

    [HttpGet("list")]
    public async Task<IActionResult> GetDocumentList()
    {
        try
        {
            var containerClient = _blobServiceClient.GetBlobContainerClient(_uploadFolder);

            if (!await containerClient.ExistsAsync())
            {
                return NotFound("UploadedDocuments container not found.");
            }

            var files = containerClient.GetBlobs().Select(blobItem => blobItem.Name).ToList();

            return Ok(new { Files = files });
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Internal server error: {ex.Message}");
        }
    }

    [HttpPost("upload")]
    public async Task<IActionResult> UploadDocument(IFormFile file)
    {
        try
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("Invalid file");
            }

            var containerClient = _blobServiceClient.GetBlobContainerClient(_uploadFolder);
            await containerClient.CreateIfNotExistsAsync();

            var originalFileName = Path.GetFileName(file.FileName);
            var blobClient = containerClient.GetBlobClient(originalFileName);

            using (var stream = file.OpenReadStream())
            {
                await blobClient.UploadAsync(stream, true);
            }

            return Ok(new { OriginalFileName = originalFileName, FileName = originalFileName, FilePath = blobClient.Uri.ToString(), FolderName = _uploadFolder });
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Internal server error: {ex.Message}");
        }
    }

    [HttpDelete("delete/{fileName}")]
    public async Task<IActionResult> DeleteDocument(string fileName)
    {
        try
        {
            var containerClient = _blobServiceClient.GetBlobContainerClient(_uploadFolder);
            var blobClient = containerClient.GetBlobClient(fileName);

            if (await blobClient.ExistsAsync())
            {
                await blobClient.DeleteAsync();
                return Ok($"Document '{fileName}' deleted successfully.");
            }
            else
            {
                return NotFound($"Document '{fileName}' not found.");
            }
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Internal server error: {ex.Message}");
        }
    }
}
