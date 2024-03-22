using Microsoft.EntityFrameworkCore;
using Query_Quasar_Bot_API.Models;

public class LoginDbContext : DbContext
{
    public LoginDbContext(DbContextOptions<LoginDbContext> options) : base(options)
    {
    }

    public DbSet<LoginRequest> Logins { get; set; } // Change this line to match the DbSet name
}
